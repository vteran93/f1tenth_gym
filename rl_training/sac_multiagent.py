# train_multiagent_sac.py

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
import torch
from collections.abc import Iterable

# --- Custom Wrapper to Expose Single Agent View ---
from stable_baselines3.common.logger import configure


logger = logging.getLogger(__name__)

class SingleAgentEnv(gym.Env):
    def __init__(self, full_env, agent_idx):
        super(SingleAgentEnv, self).__init__()
        self.full_env = full_env
        self.agent_idx = agent_idx

        # Define action and observation spaces based on one agent
        # Example starting poses (as many as num_agents in full_env)
        # Puedes parametrizar esto
        default_poses = np.array([[0., 0., 0.], [2., 0., 0.]])
        full_obs, _, _, _ = self.full_env.reset(default_poses)

        single_obs = self.extract_obs(full_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=single_obs.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float32
        )

    def reset(self):
        default_poses = np.zeros((self.full_env.num_agents, 3))
        obs, _, _, _ = self.full_env.reset(default_poses)
        return self.extract_obs(obs)


    def step(self, action):
        # In multiagent setup, this will not be used in training
        raise NotImplementedError("Use centralized stepping in train loop.")

    def extract_obs(self, full_obs):
        scan = full_obs['scans'][self.agent_idx]
        x = full_obs['poses_x'][self.agent_idx]
        y = full_obs['poses_y'][self.agent_idx]
        theta = full_obs['poses_theta'][self.agent_idx]
        linear_x = full_obs['linear_vels_x'][self.agent_idx]
        ang_z = full_obs['ang_vels_z'][self.agent_idx]
        # Concatenate into a flat observation vector
        return np.concatenate([scan, [x, y, theta, linear_x, ang_z]])

# --- Main Training Loop ---


def main():
    num_agents = 2

    # Create full environment
    full_env = gym.make(
        'f110_gym:f110-v0',
        num_agents=num_agents,
        map='../gym/f110_gym/envs/maps/vegas',
        map_ext='.png'
    )

    print("Mapa cargado:", full_env.unwrapped.map_path)

    # Create one wrapper per agent
    agent_envs = [SingleAgentEnv(full_env, idx) for idx in range(num_agents)]

    # Create a SAC model per agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    models = []

    for i, env in enumerate(agent_envs):
        # Logger individual para cada agente
        new_logger = configure(folder=f"./logs_agent_{i}", format_strings=["stdout"])
        
        # Crear modelo SAC
        model = SAC("MlpPolicy", DummyVecEnv([lambda env=env: env]), verbose=1, device=device)
        model.set_logger(new_logger)
        
        models.append(model)


    # Training loop parameters
    n_episodes = 1000
    steps_per_episode = 500

    for ep in range(n_episodes):
        # Reset environment
        poses = np.array([[0., 0., 0.], [2., 0., 0.]]
                         )  # Starting poses example
        obs_full, _, _, _ = full_env.reset(poses)

        for step in range(steps_per_episode):
            actions = []

            #full_env.render(mode='human_fast')  # Esto actualiza la visualización

            # Each agent predicts its action
            for i in range(num_agents):
                obs_agent = agent_envs[i].extract_obs(obs_full)
                #models[i].set_logger(logger)
                action, _ = models[i].predict(obs_agent, deterministic=False)
                # Garantiza shape=(1, action_dim)
                action = np.array(action).reshape(1, -1)
                actions.append(action)

            actions_array = np.vstack(actions)

            # Step full environment with all actions
            obs_full, reward, done, info = full_env.step(actions_array)

            # Store transition manually into replay buffers
            for i in range(num_agents):
                obs_agent = agent_envs[i].extract_obs(obs_full)
                models[i].replay_buffer.add(
                    obs=np.array([agent_envs[i].extract_obs(obs_full)]),      # shape (1, obs_dim)
                    next_obs=np.array([obs_agent]),                           # shape (1, obs_dim)
                    action=np.array([actions[i]]),                            # shape (1, act_dim)
                    reward=np.array([reward]),                                # shape (1,)
                    done=np.array([float(done)]),                             # shape (1,)
                    infos=[{}]                                                 # must be a list of dicts
                )

            # Train each agent
            for model in models:
                model.train(batch_size=64, gradient_steps=1)


            if done or (isinstance(done,Iterable) and all(done)):
                break

        print(f"Episode {ep+1}/{n_episodes} finished.")

    # Save all models
    for idx, model in enumerate(models):
        model.save(f"sac_agent_{idx}")


if __name__ == "__main__":
    main()
