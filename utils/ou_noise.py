"""
Code to model the Ornstein-Uhlenbeck process as noise for exploration for DDPG agents.

Based on https://github.com/greatwallet/mountain-car/blob/master/OU_Noise.py
"""

import numpy as np
import copy


class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=seed)
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.state += self.theta * (self.mu - self.state) + self.sigma * self.rng.normal(size=self.mu.shape)
        return self.state
