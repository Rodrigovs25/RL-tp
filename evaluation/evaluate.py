import gymnasium as gym
from agent.dqn_agent import DQN_Agent


def evaluate_agent(num_episodes=10):
    """
    Avalia um agente DQN NÃO TREINADO (pois não salvamos pesos).
    Apenas roda alguns episódios com baixa exploração.
    """

    # Cria o ambiente
    env = gym.make("LunarLander-v3", render_mode="human")

    # Cria um agente novo (não treinado)
    agent = DQN_Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        replay_buffer_size=1,     # não precisa de buffer
        batch_size=1,
        alpha=0.001,
        gamma=0.99,
        initial_epsilon=0.05,     # baixa exploração
        final_epsilon=0.05,
        epsilon_decay=1.0,        # sem decay
        C=500,
        train_freq=4
    )

    returns = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)  # ação sem aprendizado
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        returns.append(total_reward)
        print(f"Episode {ep+1}/{num_episodes} — Return: {total_reward:.2f}")

    env.close()
    return returns
