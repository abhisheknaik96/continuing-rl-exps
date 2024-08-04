"""
Implements an n-state loop MDP where the n states are in a loop
with a single +1 reward from the transition from state 0 to 1.
"""

from environments.base_environment import FeaturesEnv
import numpy as np


class LoopN(FeaturesEnv):
    def __init__(self, **env_args):
        """Setup for the environment called when the experiment first starts."""
        super().__init__()
        self.num_states = env_args.get('num_states', 3)
        self.num_actions = 1

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.reward_scale_factor = env_args.get('reward_scale_factor', 1)
        self.reward_offset = env_args.get('reward_offset', 0)

        # adding connections from node i to i+1
        for s in range(self.num_states - 1):
            self.P[s, 0, s + 1] = 1
        # connection from N-1th to 0th node
        self.P[self.num_states - 1, 0, 0] = 1
        # rewards for going from 0 to 1
        self.R[0, 0, 1] = 1
        self.R *= self.reward_scale_factor
        self.R += self.reward_offset

        self.rng_seed = env_args.get('rng_seed', 22)
        self.rng = np.random.RandomState(self.rng_seed)

        self.X = self.create_feature_matrix()
        self.num_features = self.X.shape[1]

        self.start_state = 0
        self.reward_obs_term = [0.0, None, False]
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.timestep = 0

    def create_feature_matrix(self):
        return np.identity(self.num_states)


def test(rng_seed=0, num_steps=100, debug=True):
    env = Loop(num_states=3, rng_seed=rng_seed,
               reward_offset=0, reward_scale_factor=3)
    obs = env.start()
    if debug:
        print(obs)
    i = 0
    rewards = np.zeros(num_steps)
    np.random.seed(rng_seed)

    while i < num_steps:
        action = 0
        reward, obs, = env.step(action)
        rewards[i] = reward
        if debug:
            print(action, reward, obs)
        i += 1

    if not debug:
        print(f'Empirical average reward of the MRP over {num_steps} steps: '
              f'{np.mean(rewards):.4f}\n')


if __name__ == "__main__":
    # test(num_steps=10000, debug=False)
    test(rng_seed=0, num_steps=20, debug=True)
