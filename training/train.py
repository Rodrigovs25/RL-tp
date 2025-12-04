import gymnasium as gym
from agent.dqn_agent import DQNAgent


def run_training(params, save_path="models/dqn_weights.pth"):
    env = gym.make("LunarLander-v3")

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        replay_buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        alpha=params["alpha"],
        gamma=params["gamma"],
        initial_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_decay=params["epsilon_decay"],
        C=params["target_update"],
        train_freq=params["train_freq"],
    )

    returns = agent.train(env, params["episodes"])
    agent.save(save_path)

    env.close()
    return returns