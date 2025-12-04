import gymnasium as gym
import torch
from agent.dqn_agent import DQN_Agent


def evaluate(model_path, episodes=10):
    env = gym.make("LunarLander-v3")

    agent = DQN_Agent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()

    results = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32)
                action = torch.argmax(agent.policy_net(state_t)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Episode {ep+1}: Return = {total_reward}")
        results.append(total_reward)

    env.close()
    return results
