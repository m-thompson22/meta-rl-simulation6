import numpy as np
import torch 

class ActorInputBuffer:
    def __init__(self, seq_len=20, num_actions=2):
        self.seq_len = seq_len
        self.num_actions = num_actions
        self.buffer = np.zeros((seq_len, num_actions + 1), dtype=np.float32)

    def update(self, action, rpe):
        one_hot_action = np.zeros(self.num_actions, dtype=np.float32)
        one_hot_action[action] = 1.0
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = np.concatenate([one_hot_action, [rpe]])

    def get_tensor(self):
        return torch.tensor(self.buffer.copy(), dtype=torch.float32).unsqueeze(0)


class CriticInputBuffer:
    def __init__(self, seq_len=20, num_actions=2):
        self.seq_len = seq_len
        self.num_actions = num_actions
        self.buffer = np.zeros((seq_len, num_actions + 1), dtype=np.float32)

    def update(self, reward, action):
        one_hot_action = np.zeros(self.num_actions, dtype=np.float32)
        one_hot_action[action] = 1.0
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = np.concatenate([one_hot_action, [reward]])

    def get_tensor(self):
        return torch.tensor(self.buffer.copy(), dtype=torch.float32).unsqueeze(0)

    def peek_next(self, reward, action):
        one_hot_action = np.zeros(self.num_actions, dtype=np.float32)
        one_hot_action[action] = 1.0
        new_buffer = self.buffer.copy()
        new_buffer = np.roll(new_buffer, shift=-1, axis=0)
        new_buffer[-1] = np.concatenate([one_hot_action, [reward]])
        return torch.tensor(new_buffer, dtype=torch.float32).unsqueeze(0)