import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.network import DQN
from utils.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN_Agent:
    def __init__(
        self, observation_space, action_space, replay_buffer_size, batch_size,
        alpha, gamma, initial_epsilon, final_epsilon, epsilon_decay, C, train_freq
    ):
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.C = C
        self.train_freq = train_freq

        # Replay buffer
        self.memory = ReplayBuffer(replay_buffer_size)

        # Networks on GPU
        in_dim = observation_space.shape[0]
        out_dim = action_space.n

        self.policy_net = DQN(in_dim, out_dim).to(DEVICE)
        self.target_net = DQN(in_dim, out_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.SmoothL1Loss()

    # =============================
    #  EPSILON-GREEDY ACTION
    # =============================
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return np.random.randint(self.action_space.n)

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(state)
        return q.argmax().item()

    # =============================
    #  LEARNING STEP (GPU)
    # =============================
    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # =============================
    #  TRAIN LOOP (optimized)
    # =============================
    def train(self, env, episodes):

        rewards = []
        total_steps = 0

        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            ep_reward = 0

            while not (done or truncated):
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                self.memory.add(state, action, reward, next_state, done)

                state = next_state
                ep_reward += reward

                # TRAIN
                if total_steps % self.train_freq == 0 and len(self.memory) > self.batch_size:
                    self.learn()

                # TARGET UPDATE
                if total_steps % self.C == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                total_steps += 1

            # EPSILON DECAY
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

            rewards.append(ep_reward)

        return rewards
