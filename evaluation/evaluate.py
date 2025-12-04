import gymnasium as gym
from agent.dqn_agent import DQN_Agent


def evaluate_agent(num_episodes=10):
    """
    Avaliação simples: cria um agente novo e roda episódios sem treinamento.
    """

    env = gym.make("LunarLander-v3", render_mode="human")

    agent = DQN_Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        replay_buffer_size=1,  
        batch_size=1,           
        alpha=0.001,
        gamma=0.99,
        initial_epsilon=0.0,    
        final_epsilon=0.0,
        epsilon_decay=1.0,
        C=500,
        train_freq=4
    )

    returns = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):

            action = agent.choose_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward

        returns.append(ep_reward)
        print(f"Episode {ep+1}/{num_episodes} - Return: {ep_reward:.2f}")

    env.close()
    print("\nEvaluation finished.")
    return returns
