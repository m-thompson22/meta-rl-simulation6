import numpy as np
from core.config import Config

class RiskySafeTask:
    def __init__(self, block_schedule, arm_map=None, block_type="control"):
        self.block_schedule = sorted(block_schedule, key=lambda x: x[0])
        self.current_p_risky = block_schedule[0][1]
        self.rS = 1
        self.rL = 4
        self.block_type = block_type
        self.arm_map = arm_map if arm_map is not None else {0: "safe", 1: "risky"}

    def update_probability(self, trial_num):
      for t, p in reversed(self.block_schedule):
        if trial_num >= t:
            self.current_p_risky = p
            break

    def get_reward(self, action):
      arm = self.arm_map[action]
      if arm == "safe":
        reward = self.rS
      else:  # risky
        reward = self.rL if np.random.rand() < self.current_p_risky else 0
      return reward, arm
