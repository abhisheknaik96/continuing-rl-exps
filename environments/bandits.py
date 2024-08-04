"""
Implements a multi-armed bandit testbed 
like the one in Sutton and Barto's (2018) Section 2.3.
The rewards are stationary (for now).
"""

import numpy as np


class MultiArmedBandit():
    def __init__(self, **env_args):
        """Setup for the environment called when the experiment first starts."""
        super().__init__()
        self.num_states = 1
        self.num_actions = env_args.get('num_actions', 10)

        self.rng_seed = env_args.get('rng_seed', 22)
        self.rng = np.random.default_rng(self.rng_seed)
        self.R = self.rng.normal(loc=0, scale=1, size=self.num_actions)
        self.best_action = np.argmax(self.R)
        self.reward_scale_factor = env_args.get('reward_scale_factor', 1)
        self.reward_offset = env_args.get('reward_offset', 0)
        self.R = (self.R * self.reward_scale_factor) + self.reward_offset
        self.sampled_reward_stddev = env_args.get('sampled_reward_stddev', 1)

        self.state = np.array([1])
        self.reward_obs_term = [0.0, None, False]
        self.best_action_count = 0

    def start(self):
        """The first method called when the episode starts, called before the
        agent starts."""
        return self.state

    def step(self, action):
        """A step taken by the environment.

        Args:
            action: the action taken by the agent

        Returns:
            a tuple of the reward and observation
        """
        if action == self.best_action:
            self.best_action_count += 1
        reward = self.rng.normal(self.R[action], self.sampled_reward_stddev)
        self.reward_obs = (reward, self.state)
        return self.reward_obs


def test(rng_seed=0, num_steps=100, debug=True, num_actions=10):
    env = MultiArmedBandit(num_actions=num_actions, rng_seed=rng_seed, reward_offset=10)
    print(f"Rewards: {env.R}")
    obs = env.start()
    if debug:
        print(obs)
    i = 0
    rewards = np.zeros(num_steps)
    np.random.seed(rng_seed)

    while i < num_steps:
        action = np.random.randint(num_actions)
        reward, obs, = env.step(action)
        rewards[i] = reward
        if debug:
            print(action, reward, obs)
        i += 1

    print(f'Empirical average reward of the MRP over {num_steps} steps: '
            f'{np.mean(rewards):.4f}\n')


if __name__ == "__main__":
    # test(rng_seed=0, num_steps=1000000, debug=False)
    test(rng_seed=0, num_steps=30, debug=True, num_actions=10)
