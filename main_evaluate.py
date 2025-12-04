from evaluation.evaluate import evaluate_agent

if __name__ == "__main__":
    returns = evaluate_agent(num_episodes=10)
    print("Returns:", returns)
