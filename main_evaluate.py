from evaluation.evaluate import evaluate_agent

if __name__ == "__main__":
    print("Starting evaluation...")
    returns = evaluate_agent(num_episodes=10)
    print("Evaluation finished!")
    print("Returns:", returns)
