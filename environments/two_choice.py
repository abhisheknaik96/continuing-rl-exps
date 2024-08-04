"""
Implements a variant of the environment from Mahadevan (1994),
also used in this form by Naik et al. (2019) (https://arxiv.org/abs/1910.02140).
"""

from environments.base_environment import FeaturesEnv
import numpy as np


class TwoChoiceMDP(FeaturesEnv):
    def __init__(self, **env_args):
        """Setup for the environment called when the experiment first starts."""
        super().__init__()
        self.num_states = 9
        self.num_actions = 2

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.reward_scale_factor = env_args.get('reward_scale_factor', 1)
        self.reward_offset = env_args.get('reward_offset', 0)

        # adding connections from node i to i+1
        for s in range(self.num_states - 1):
            self.P[s, 0, s + 1] = 1
            self.P[s, 1, s + 1] = 1
        # connection from N-1th to 0th node
        self.P[8, 0, 0] = 1
        self.P[8, 1, 0] = 1
        # removing the connection from 4th to 5th node
        self.P[4, 0, 5] = 0
        self.P[4, 1, 5] = 0
        # connection from 4th to 0th node
        self.P[4, 0, 0] = 1
        self.P[4, 1, 0] = 1
        # action 1 in node 0 should not lead to 1, but 5
        self.P[0, 1, 1] = 0
        self.P[0, 1, 5] = 1
        # rewards for going from 0 to 1 and 8 to 0
        self.R[0, 0, 1] = 1
        self.R[8, 0, 0] = 2
        self.R[8, 1, 0] = 2
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
    env = TwoChoiceMDP(rng_seed=rng_seed, reward_offset=-1)
    obs = env.start()
    if debug:
        print(obs)
    i = 0
    rewards = np.zeros(num_steps)
    np.random.seed(rng_seed)

    while i < num_steps:
        action = np.random.choice(2)
        reward, obs, = env.step(action)
        rewards[i] = reward
        if debug:
            print(action, reward, obs)
        i += 1

    if not debug:
        print(f'Empirical average reward of the MRP over {num_steps} steps: '
              f'{np.mean(rewards):.4f}\n')


if __name__ == "__main__":
    test(num_steps=10000, debug=False)
    # test(rng_seed=0, num_steps=20, debug=True)
