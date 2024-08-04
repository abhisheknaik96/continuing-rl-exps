"""Implements the classic RiverSwim environment."""


from environments.base_environment import FeaturesEnv
import numpy as np


class RiverSwim(FeaturesEnv):
    """Implements the environment for an RLGlue environment
    """

    def __init__(self, **env_args):
        """Setup for the environment called when the experiment first starts."""
        super().__init__()
        self.num_states = 6
        self.num_actions = 2

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.reward_scale_factor = env_args.get('reward_scale_factor', 1)
        self.reward_offset = env_args.get('reward_offset', 0)

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

        for s in range(self.num_states):
            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.7
                self.P[s, 1, s + 1] = 0.3
                self.R[s, 0, s] = 5.0 / 10000
            elif s == self.num_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.7
                self.P[s, 1, s] = 0.3
                self.R[s, 1, s] = 1
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.1
                self.P[s, 1, s] = 0.6
                self.P[s, 1, s + 1] = 0.3
        self.R *= self.reward_scale_factor
        self.R += self.reward_offset

        self.rng_seed = env_args.get('rng_seed', 22)
        self.rng = np.random.RandomState(self.rng_seed)

        self.X = self.create_feature_matrix()
        self.num_features = self.X.shape[1]

        # self.start_state = self.rng.choice(self.num_states)
        self.start_state = 0
        self.reward_obs_term = [0.0, None, False]
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.timestep = 0

    def create_feature_matrix(self):
        return np.identity(self.num_states)


def test(rng_seed=0, num_steps=100, debug=True, reward_scale_factor=1):
    env = RiverSwim(rng_seed=rng_seed,
                    reward_offset=0,
                    reward_scale_factor=reward_scale_factor)
    obs = env.start()
    if debug:
        print(obs)
    i = 0
    rewards = np.zeros(num_steps)
    np.random.seed(rng_seed)
    r_bar = 0

    while i < num_steps:
        # action = np.random.randint(0, env.num_actions)
        action = 1
        reward, obs, = env.step(action)
        rewards[i] = reward
        beta = 0.03
        r_bar += beta * (reward - r_bar)
        if debug:
            print(action, reward, obs)
        i += 1
        beta *= 0.9995

    if not debug:
        print(f'Empirical average reward of the MRP over {num_steps} steps: '
              f'{np.mean(rewards):.4f}\n')
        print(r_bar)


if __name__ == "__main__":
    test(num_steps=10000, debug=False, reward_scale_factor=10)
    # test(rng_seed=2, num_steps=20, debug=True)
