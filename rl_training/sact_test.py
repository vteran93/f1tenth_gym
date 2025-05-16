import gym
import numpy as np
from stable_baselines3 import SAC
import torch
from sac_multiagent import SingleAgentEnv  # Usa la misma clase wrapper


def main():
    num_agents = 2
    full_env = gym.make(
        'f110_gym:f110-v0',
        num_agents=num_agents,
        map='../gym/f110_gym/envs/maps/vegas',
        map_ext='.png'
    )

    agent_envs = [SingleAgentEnv(full_env, idx) for idx in range(num_agents)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating using {device} device")

    models = []
    for i in range(num_agents):
        model = SAC.load(f"sac_agent_{i}", device=device)
        models.append(model)

    # Evaluación (puedes repetir varias veces si deseas promediar métricas)
    obs_full, _, _, _ = full_env.reset(np.array([[0., 0., 0.], [2., 0., 0.]]))
    done = False
    step = 0
    max_steps = 500

    while not done and step < max_steps:
        full_env.render(mode='human_fast')  # Visualización (si tienes display)
        actions = []

        for i in range(num_agents):
            obs_agent = agent_envs[i].extract_obs(obs_full)
            action, _ = models[i].predict(obs_agent, deterministic=True)
            actions.append(np.array(action).reshape(1, -1))

        actions_array = np.vstack(actions)
        obs_full, reward, done, info = full_env.step(actions_array)

        if isinstance(done, (list, tuple, np.ndarray)) and all(done):
            break
        step += 1

    print("Evaluation finished after", step, "steps.")


if __name__ == "__main__":
    main()
