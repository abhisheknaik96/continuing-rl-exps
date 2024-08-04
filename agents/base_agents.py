"""Code for prediction agents."""

import math
import numpy as np
from utils.helpers import get_weights_from_npy
from utils.tilecoder import TileCoder


class LFAPredictionAgent:
    """
    A generic class that is re-used in the implementation of all sorts of
    prediction algorithms with LFA.
    In most cases, only agent_step needs to be implemented in the child classes.
    """

    def __init__(self, **agent_args):
        """
        Initializes the agent-class parameters.

        Args:
            **agent_args: a dictionary of agent parameters
            num_actions: number of discrete actions
            num_features: size of the observation vector
            rng_seed: seed for the random-number generator
            alpha: the step size for updating the weights
            eta: the step-size parameter for updating the reward-rate estimate
            alpha_sequence: sequence of step size: 'exponential' or '1byN'
            alpha_decay_rate: decay rate if sequence is exponential(ly decaying)
            value_init: optional initial values
            avg_reward_init: optional initial value of the reward-rate estimate
            weights: the vector of weights of size self.num_features
            avg_reward: the scalar reward-rate estimate
            pi: the target policy
            b: the behavior policy
        """
        super().__init__()
        assert 'num_actions' in agent_args

        self.approximation_type = agent_args.get('approximation_type', 'linear')
        assert self.approximation_type in ['tabular', 'linear']

        self.num_actions = agent_args['num_actions']
        assert 'num_features' in agent_args
        self.num_features = agent_args['num_features']

        self.rng_seed = agent_args.get('rng_seed', 22)
        self.rng = np.random.default_rng(seed=self.rng_seed)

        self.alpha = agent_args.get('alpha', 0.001)
        self.alpha_sequence = agent_args.get('alpha_sequence', 'exponential')
        if self.alpha_sequence == 'exponential':
            self.alpha_decay_rate = agent_args.get('alpha_decay_rate', 1.0)
            self.alpha_init = self.alpha
        self.value_init = agent_args.get('value_init', 0)

        self.bias = agent_args.get('bias', False)
        if self.approximation_type == 'linear':
            self.bias = True
        if self.bias:
            self.num_features += 1

        self.weights = np.zeros(self.num_features) + self.value_init
        # if the weights are supplied, they should be an ndarray in an npy file
        # as a dictionary element with key 'weights'
        if 'weights_file' in agent_args:
            weights = get_weights_from_npy(agent_args['weights_file'])
            assert weights.size == self.num_features * self.num_actions
            self.weights = weights

        assert 'pi' in agent_args
        self.pi = agent_args['pi']
        self.b = agent_args.get('b', self.pi)
        self._validate_behavioural_policy()

        self.actions = list(range(self.num_actions))
        self.past_action = None
        self.past_state = None
        self.past_rho = None
        self.timestep = 0

    def _choose_action(self, state):
        """Returns an action index based on the behaviour policy.
        For now the policy is observation-agnostic.

        Args:
            state: the current agent state
        Returns:
            action: the selected action's index
        """
        action = self.rng.choice(self.actions, p=self.b)
        return action

    def _get_value(self, state):
        """Returns the linear estimation of the value for a given feature vector

        Args:
            state: the agent-state vector
        Returns:
            value: w^T x
        """
        value = np.dot(state, self.weights)
        return value

    def _validate_behavioural_policy(self):
        okay = 1
        for i, _ in enumerate(self.pi):
            if self.pi[i] > 0 and self.b[i] == 0:
                okay = 0
                break
        assert okay == 1, 'Behavioural policy does not have coverage'

    def _initialize_step_size(self):
        """Initializes the step size."""
        if self.alpha_sequence == 'exponential':
            self.alpha = self.alpha_init
        elif self.alpha_sequence == '1byN':
            self.alpha = 1

    def _update_step_size(self):
        """Updates the step size per step."""
        if self.alpha_sequence == 'exponential':
            self.alpha *= self.alpha_decay_rate
        elif self.alpha_sequence == '1byN':
            self.alpha = 1 / self.timestep

    def _get_representation(self, observation, action=-1):
        """Returns the agent state.

        This is a simple agent-state function that returns
        different representations for state-value and action-value functions.
        When using action-value functions, pass the (discrete) action index
        for which the representation is required. The function returns a
        one-hot version of the representation vector.
        For instance, ([1,2], 1) returns [0,0,1,2,0,0] for self.num_actions=3.
        When using state-value functions, pass action=-1 to get back
        the observation as is.
        For instance, ([1,2],-1) returns [1,2].

        Args:
            observation: the current observation
            action: the action index
        Returns:
            rep: the agent-state vector
        """
        if self.bias:
            observation = np.concatenate((observation, [1]))
        if action == -1:
            rep = observation
        else:
            rep = np.zeros(self.num_features * self.num_actions)
            idx_start = self.num_features * action
            idx_end = self.num_features * (action + 1)
            rep[idx_start:idx_end] = observation
        return rep

    def start(self, observation):
        """
        The first method called when the experiment starts, called after the
        environment starts.

        Args:
            observation: the observation from the environment's env_start()
        Returns:
            action: the first action the agent takes
        """
        self._initialize_step_size()
        state = self._get_representation(observation, -1)
        action = self._choose_action(state)
        self.timestep += 1

        self.past_action = action
        self.past_state = state
        # ToDo: very hacky, specific to TwoChoice
        # self.past_rho = self.pi[action] / self.b[action] if state[0] == 1 else 1.0
        self.past_rho = self.pi[action] / self.b[action]

        return action

    def step(self, reward, observation):
        """
        Returns a new action corresponding to the new observation
        and updates the agent parameters.

        Args:
            reward: the reward at the current time step
            observation: the new observation vector
        Returns:
            action: an action-index integer
        """
        state = self._get_representation(observation, -1)
        self._update_weights(reward, state)
        action = self._choose_action(state)

        self.timestep += 1
        self._update_step_size()

        # ToDo: very hacky, specific to TwoChoice
        # self.past_rho = self.pi[action] / self.b[action] if state[0] == 1 else 1.0
        self.past_rho = self.pi[action] / self.b[action]
        self.past_state = state
        self.past_action = action

        return action

    def _update_weights(self, reward, state):
        """
        Performs the update using S_t, A_t, R_{t+1}, S_{t+1}.
        Every agent has to implement this method.
        Args:
            reward: R_{t+1}
            state: S_{t+1}

        Returns:
            Nothing
        """
        return NotImplementedError


class LFAControlAgent:
    """
    A generic class that is re-used in the implementation of all sorts of
    control algorithms with LFA.
    In most cases, only agent_step needs to be implemented in the child classes.
    """

    def __init__(self, **agent_args):
        """
        Initializes the agent-class parameters.

        Args:
            **agent_args: a dictionary of agent parameters
            num_actions: number of discrete actions
            num_features: size of the observation vector
            rng_seed: seed for the random-number generator
            alpha: the step size for updating the weights
            eta: the step-size parameter for updating the reward-rate estimate
            alpha_sequence: sequence of step size: 'exponential' or '1byN'
            alpha_decay_rate: decay rate if sequence is exponential(ly decaying)
            value_init: optional initial values
            avg_reward_init: optional initial value of the reward-rate estimate
            weights: the vector of weights of size self.num_features
            avg_reward: the scalar reward-rate estimate
            pi: the target policy
            b: the behavior policy
        """
        # input the function approximator
        self.approximation_type = agent_args.get('approximation_type', 'linear')
        assert self.approximation_type in ['tabular', 'linear']

        # take the environment parameters as input
        assert 'num_actions' in agent_args
        self.num_actions = agent_args['num_actions']
        assert 'num_features' in agent_args
        self.num_features = agent_args['num_features']

        # initialize the random-number generator
        self.rng_seed = agent_args.get('rng_seed', 22)
        self.rng = np.random.default_rng(seed=self.rng_seed)

        # initialize the step size and optional step-size sequence
        self.alpha_init = agent_args.get('alpha', 0.001)
        self.alpha_sequence = agent_args.get('alpha_sequence', 'exponential')
        self.robust_to_initialization = agent_args.get('robust_to_initialization', False)
        # if self.robust_to_initialization:
        #     self.alpha_sequence = 'unbiased_trick'
        if self.alpha_sequence == 'exponential':
            self.alpha_decay_rate = agent_args.get('alpha_decay_rate', 1.0)

        # initialize the tile coder, if any
        self.tilecoder = agent_args.get("tilecoder", False)
        if self.tilecoder:
            self.num_tilings = agent_args.get("num_tilings", 8)
            assert "tiling_dims" in agent_args
            tiling_dims = agent_args["tiling_dims"]
            assert "limits_per_dim" in agent_args
            limits_per_dim = agent_args["limits_per_dim"]

            self.tilecoder = TileCoder(tiling_dims=tiling_dims,
                                       limits_per_dim=limits_per_dim,
                                       num_tilings=self.num_tilings,
                                       style='indices')

            self.num_features = self.tilecoder.n_tiles
            self.alpha_init /= self.num_tilings

        if self.approximation_type == 'linear':
            self.bias = True
            self.num_features += 1
        else:
            self.bias = False

        self.weights = np.zeros(self.num_features * self.num_actions)
        self.avg_reward = None

        # if the weights are supplied, they should be an ndarray in an npy file
        # as a dictionary element with key 'weights'
        if 'weights_file' in agent_args:
            weights = get_weights_from_npy(filename=agent_args['weights_file'],
                                           seed_idx=agent_args['load_seed'])
            assert weights.size == self.num_features * self.num_actions
            self.weights = weights

        # initialize the e-greedy exploration parameters
        self.epsilon_start = agent_args.get('epsilon_start', 0.9)
        self.epsilon_end = agent_args.get('epsilon_end', 0.05)
        self.epsilon_decay_param = agent_args.get('epsilon_decay_param', 200000)
        self.epsilon = self.epsilon_start

        self.actions = list(range(self.num_actions))
        self.past_action = None
        self.past_obs = None
        self.exploratory_action = None
        self.timestep = 0
        self.update_count = 1
        self.max_value_per_step = None

    def _process_raw_observation(self, obs):
        """
        Processes raw observation into a encoding, e.g., a tile-coded encoding.
        """
        if self.tilecoder:
            observation = self.tilecoder.getitem(obs)
            # tilecoder observations are indices, so add the index of the bias
            observation = np.concatenate((observation, [self.num_features - 1]))
        else:
            observation = obs
            if self.bias:
                observation = np.concatenate((observation, [1]))

        return observation

    def _get_representation(self, observation, action):
        """Returns the agent state.

        This simple implementation of the agent-state-update function returns
        a one-hot version of the observation vector.
        For instance, ([1,2], 1) returns [0,0,1,2,0,0] for self.num_actions=3.

        Args:
            observation: the current observation
            action: the action index
        Returns:
            state: the agent-state vector
        """
        if self.tilecoder:
            offset = self.num_features * action
            state = np.array(observation + offset, dtype=int)
        else:
            state = np.zeros(self.num_features * self.num_actions)
            offset_start = self.num_features * action
            offset_end = self.num_features * (action + 1)
            state[offset_start:offset_end] = observation

        return state

    def _get_linear_value_approximation(self, representation):
        """Returns the linear estimation of the value for a given representation

        Args:
            representation: the representation vector
        Returns:
            value: w^T x
        """
        if self.tilecoder:      # assumes 'indices' style for tilecoder
            value = np.sum(self.weights[representation])
        else:
            value = np.dot(representation, self.weights)
        return value

    def _get_value(self, observation, action):
        """Returns the value corresponding to an observation and action.

        Args:
            observation: the observation vector
            action: the action index
        Returns:
            value: an approximation of the expected value
        """
        rep = self._get_representation(observation, action)
        return self._get_linear_value_approximation(rep)

    def _set_weights(self, given_weight_vector):
        """Sets the agent's weights to the given weight vector."""
        self.weights = given_weight_vector

    def _argmax(self, values):
        """Returns the argmax of a list. Breaks ties uniformly randomly."""
        self.max_value_per_step = np.max(values)
        return self.rng.choice(np.argwhere(values == self.max_value_per_step).flatten())

    def _choose_action_egreedy(self, state):
        """
        Returns an action using an epsilon-greedy policy w.r.t. the
        current action-value function.

        Args:
            state: the current agent state
        Returns:
            action: an action-index integer
        """
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
            self.max_value_per_step = None
            self.exploratory_action = True
        else:
            q_s = np.array([self._get_value(state, a) for a in self.actions])
            action = self._argmax(q_s)
            self.exploratory_action = False
            self.update_count += 1

        return action

    def _max_action_value(self, state):
        """
        Returns the action index corresponding to the maximum action value
        for a given agent-state vector. If the maximum action value is
        shared by more than one action, one of them is randomly chosen.
        """
        q_s = np.array([self._get_value(state, a) for a in self.actions])
        argmax_action = self._argmax(q_s)

        return q_s[argmax_action]

    def start(self, observation):
        """
        The first method called when the experiment starts,
        called after the environment starts.

        Args:
            observation: the first observation returned by the environment
        Returns:
            action: an action-index integer
        """
        obs = self._process_raw_observation(observation)
        action = self._choose_action_egreedy(obs)

        self.past_obs = obs
        self.past_action = action
        self.timestep += 1
        self._initialize_step_size()

        return action

    def step(self, reward, observation):
        """
        Returns a new action corresponding to the new observation
        and updates the agent parameters.

        Args:
            reward: the reward at the current time step
            observation: the new observation vector
        Returns:
            action: an action-index integer
        """
        obs = self._process_raw_observation(observation)
        self._update_weights(reward, obs)
        action = self._choose_action_egreedy(obs)

        self.timestep += 1
        self._update_epsilon()
        self._update_step_size()

        self.past_obs = obs
        self.past_action = action

        return action

    def _initialize_step_size(self):
        """Initializes the step size."""
        if self.alpha_sequence == 'unbiased_trick':
            self.o = self.alpha_init
            self.alpha = self.alpha_init / self.o
        elif self.alpha_sequence == '1byN':
            self.alpha = 1
        else:
            self.alpha = self.alpha_init

    def _update_step_size(self):
        """Updates the step size per step."""
        if self.alpha_sequence == 'exponential':
            self.alpha *= self.alpha_decay_rate
        elif self.alpha_sequence == '1byN':
            self.alpha = 1 / self.timestep
        elif self.alpha_sequence == 'unbiased_trick':
            self.o = self.o + self.alpha_init * (1 - self.o)
            self.alpha = self.alpha_init / self.o

    def _update_epsilon(self):
        """Decays the epsilon parameter per step."""
        self.epsilon = (self.epsilon_end +
                        (self.epsilon_start - self.epsilon_end) * math.e **
                        (-self.timestep / self.epsilon_decay_param))

    def _update_weights(self, reward, obs):
        raise NotImplementedError
