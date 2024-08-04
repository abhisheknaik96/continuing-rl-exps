"""Implements the RandomWalkN environment."""

import numpy as np
from environments.base_environment import FeaturesEnv


class RandomWalkN(FeaturesEnv):
    """
    Implements a random-walk environment with N=2i+1 states.
    The state in the middle is state 0, there are i states on either side.
    From states k in (-i,i), there is a 50% probability of transitioning to
    either of its two neighbors. From state -i and i there is a deterministic
    transition to state 0 with rewards -r and r respectively.
    The MRP starts in state 0 and continues forever.
    The centered value function of state k is k*(r/i), where k in [-i,i].
    """
    def __init__(self, **env_args):
        """
        Setup for the environment; called when the experiment first starts.
        """
        super().__init__()
        self.num_states = env_args.get('num_states', 19)
        assert self.num_states % 2 == 1, "num_states should be an odd number"
        mid = self.num_states // 2
        self.num_actions = 2
        self.r_right = env_args.get('r_right', 1)
        assert self.r_right > 0, \
            'end_reward_magnitude should be a positive number'
        self.r_left = env_args.get('r_left', -self.r_right)

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.reward_scale_factor = env_args.get('reward_scale_factor', 1)
        self.reward_offset = env_args.get('reward_offset', 0)

        for s in range(1, self.num_states - 1):
            self.P[s, 0, s - 1] = 1
            self.P[s, 1, s + 1] = 1
        self.P[0, 0, mid] = 1
        self.P[0, 1, 1] = 1
        self.R[0, 0, mid] = self.r_left
        self.P[self.num_states - 1, 1, mid] = 1
        self.P[self.num_states - 1, 0, self.num_states - 2] = 1
        self.R[self.num_states - 1, 1, mid] = self.r_right

        self.R *= self.reward_scale_factor
        self.R += self.reward_offset

        self.rng_seed = env_args.get('rng_seed', 22)
        self.rng = np.random.default_rng(self.rng_seed)

        self.feature_type = env_args.get('feature_type', 'one-hot')
        assert self.feature_type in ['one-hot', 'constructed']
        self.X = self.create_feature_matrix()
        self.num_features = self.X.shape[1]

        self.start_state = mid
        self.reward_obs_term = [0.0, None, False]
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.timestep = 0

    def create_feature_matrix(self):
        """Creates the feature matrix X.

        Choices of feature_type:
        'one-hot': a matrix of one-hot feature vectors, one per state
        'constructed': a matrix of arbitrarily constructed features

        Returns: a matrix [num_states, num_features]
        """
        all_features = None
        if self.feature_type == 'one-hot':
            all_features = np.identity(self.num_states)
        elif self.feature_type == 'constructed':
            num_features = self.num_states - 1
            features = np.identity(num_features)
            last = np.zeros((1, num_features))
            last[0][self.rng.choice(num_features)] = 1
            all_features = np.concatenate((features, last), axis=0)
            self.rng.shuffle(all_features)
        return all_features


def test(rng_seed=0, num_steps=100, debug=True):
    env = RandomWalkN(num_states=5,
                      end_reward_magnitude=1,
                      rng_seed=rng_seed,
                      feature_type='one-hot')
    obs = env.start()
    i = 0
    rewards = np.zeros(num_steps)
    if debug:
        print(f'All features: {env.X}\n')
        print(obs)
    np.random.seed(rng_seed)
    # print(env.P)

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
