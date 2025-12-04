import random
import numpy as np
import torch
from collections import deque, namedtuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.experience = namedtuple(
            "Experience",
            ["state", "action", "reward", "next_state", "done"]
        )

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)

        states      = torch.tensor(np.array([e.state      for e in batch]), dtype=torch.float32, device=DEVICE)
        actions     = torch.tensor(np.array([e.action     for e in batch]), dtype=torch.int64, device=DEVICE).unsqueeze(1)
        rewards     = torch.tensor(np.array([e.reward     for e in batch]), dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.array([e.next_state for e in batch]), dtype=torch.float32, device=DEVICE)
        dones       = torch.tensor(np.array([e.done       for e in batch]), dtype=torch.float32, device=DEVICE).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

