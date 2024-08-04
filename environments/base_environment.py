"""Base agent class"""


class FeaturesEnv:
    """
    A template env that returns feature vectors as the state representation.

    Typical attributes:
        env_args: a dictionary of parameters to set up the environment
        num_states: number of states in the environment
        num_actions: number of discrete actions available in the environment
        num_features: number of features
        P: the transition probability matrix [S x A x S]
        R: the reward matrix [S x A x S]
        X: the feature matrix [S x num_features]
        feature_type: 'one_hot' or 'constructed'
        rng_seed: the seed of the random number generator
    """
    def __init__(self, **env_args):
        """Setup for the environment called when the experiment first starts.

        Mainly:
        - Set up the matrices for transition and reward dynamics
        - Construct the matrix of feature vectors
        - Initialize a tuple with the reward, first state observation, boolean
        indicating if it's terminal.
        """
        self.num_states = None
        self.num_actions = None
        self.num_features = None

        self.P = None
        self.R = None
        self.X = None  # matrix of feature vectors
        self.feature_type = None

        self.start_state = None
        self.current_state = None
        self.reward_obs = None

        self.rng_seed = None
        self.rng = None
        self.timestep = None
        self.counts = None

    def create_feature_matrix(self):
        raise NotImplementedError

    def get_features(self, state_idx):
        return self.X[state_idx]

    def start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.current_state = self.start_state
        self.reward_obs = (0, self.get_features(self.current_state))
        self.timestep += 1

        return self.reward_obs[1]

    def step(self, action):
        """A step taken by the environment.

        Args:
            action: the action taken by the agent

        Returns:
            a tuple of the reward and observation
        """
        self.counts[self.current_state, action] += 1
        cond_p = self.P[self.current_state][action]
        next_state = self.rng.choice(self.num_states, 1, p=cond_p).item()
        assert cond_p[next_state] > 0.

        reward = self.R[self.current_state, action, next_state]
        self.current_state = next_state

        self.reward_obs = (reward, self.get_features(self.current_state))
        self.timestep += 1

        return self.reward_obs
