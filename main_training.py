from training.train import run_training

if __name__ == "__main__":

    params = {
    'alpha': 0.00017195082231670288,
    'gamma': 0.9778366856839303,
    'batch_size': 128,
    'buffer_size': 50000,
    'epsilon_decay': 0.9990115359881433,
    'target_update': 500,
    'train_freq': 4,
    'episodes': 2000
}


    returns = run_training(params)
    print("Training finished!")
    # print(returns)
