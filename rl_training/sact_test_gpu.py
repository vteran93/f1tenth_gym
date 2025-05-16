import gym
import numpy as np
from stable_baselines3 import SAC
import torch
from sac_multiagent import SingleAgentEnv
import time


def render_callback(env_renderer):
    e = env_renderer
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800


def main():
    num_agents = 2
    full_env = gym.make(
        'f110_gym:f110-v0',
        num_agents=num_agents,
        map='../gym/f110_gym/envs/maps/vegas',
        map_ext='.png',
        timestep=0.001
    )
    full_env.add_render_callback(render_callback)

    agent_envs = [SingleAgentEnv(full_env, idx) for idx in range(num_agents)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating using {device} device")

    models = [SAC.load(f"sac_agent_{i}", device=device)
              for i in range(num_agents)]
    policy_nets = [model.policy.actor.to(device) for model in models]

    obs_full, _, _, _ = full_env.reset(np.array([[0., 0., 0.], [2., 0., 0.]]))
    done = False
    step = 0
    max_steps = 500
    steps_time = []

    while not done and step < max_steps:
        if step % 10 == 0:
            full_env.render(mode='human')

        obs_batch = []
        for i in range(num_agents):
            obs_np = agent_envs[i].extract_obs(obs_full)
            if np.isnan(obs_np).any() or np.isinf(obs_np).any():
                print(f"[ERROR] Obs for agent {i} has NaN or Inf:", obs_np)
                obs_np = np.nan_to_num(obs_np, nan=0.0, posinf=0.0, neginf=0.0)
            obs_batch.append(obs_np)

        obs_tensor = torch.tensor(
            np.vstack(obs_batch), dtype=torch.float32, device=device)

        start = time.time()
        action_tensor = []
        with torch.no_grad():
            for i in range(num_agents):
                obs_tensor_i = obs_tensor[i].unsqueeze(0)
                try:
                    action_i, _ = models[i].predict(
                        obs_tensor_i.cpu().numpy(), deterministic=True)
                    action_i = torch.tensor(
                        action_i, dtype=torch.float32, device=device).reshape(1, -1)
                except Exception as e:
                    print(
                        f"[ERROR] Failed to predict action for agent {i}: {e}")
                    action_i = torch.zeros(
                        (1, agent_envs[i].action_space.shape[0]), device=device)
                if torch.isnan(action_i).any():
                    print(
                        f"[ERROR] Action from agent {i} is NaN even after fallback:", action_i)
                    action_i = torch.zeros_like(action_i)
                action_tensor.append(action_i)

        action_tensor = torch.cat(action_tensor, dim=0)
        elapsed_time = time.time() - start
        steps_time.append(elapsed_time)

        actions_array = action_tensor.cpu().numpy().reshape((num_agents, -1))
        obs_full, reward, done, info = full_env.step(actions_array)
        time.sleep(0.001)

        if isinstance(done, (list, tuple, np.ndarray)) and all(done):
            break
        step += 1

    print("Evaluation finished after", step, "steps.")
    print('Average predict time:', np.average(steps_time))
    print('Mean predict time:', np.mean(steps_time))

    input('Press enter to exit')


if __name__ == "__main__":
    main()
