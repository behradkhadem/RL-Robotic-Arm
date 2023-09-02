# from normalize_action import 
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MapActionWrapper(gym.ActionWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low 
        self.high = high
        self.action_space = spaces.Box(low=low, high=high, shape=env.action_space.shape, dtype=np.float32)

    def action(self, action):
        orig_low = self.env.action_space.low
        orig_high = self.env.action_space.high
        mapped_action = self.low + (action - orig_low) * (self.high - self.low) / (orig_high - orig_low)
        return mapped_action