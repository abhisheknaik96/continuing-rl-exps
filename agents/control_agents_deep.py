"""Implements the deep versions of the control agents."""

import math
import random
import copy
from collections import deque
import numpy as np
import torch
from torch import nn
from utils.helpers import validate_output_folder


def build_fc_net_tanh(layer_sizes):
    """Returns a full-connected network with tanh activation functions."""
    assert len(layer_sizes) > 1
    layers = []
    for index in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
        act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
        layers += (linear, act)
    return nn.Sequential(*layers)


class DeepCDiscAgent:
    """Base class for the deep discounted control agents."""
    def __init__(self, **agent_args):
        """
        Args:
            seed: seed for random number generator
            epsilon: the parameter for epsilon-greedy action selection 
            epsilon_decay: the rate at which epsilon is exponentially decayed
            layer_sizes: the size of the action-value network
            net_sync_freq: frequency of syncing target network with main network  
            load_model_from: location if loading saved model
            param_update_freq: frequency of making updates
            step_size: the step size for the value estimates
            buffer_size: size of the buffer 
            batch_size: size of the mini-batch
            device: cpu or gpu 
            save_model_loc: location to save the model
        """
        assert 'device' in agent_args, "device needs to be specified in agent_args"
        self.device = agent_args['device']

        # initializing the RNG seed for reproducibility
        self.seed = agent_args.get('rng_seed', 42)
        torch.manual_seed(self.seed)
        random.seed(self.seed)  # ToDo: don't like this. Remove need for this package

        # initializing the parameters for epsilon-greedy action selection
        self.epsilon_start = torch.tensor(agent_args.get('epsilon_start', 0.9)).float().to(self.device)
        self.epsilon_end = torch.tensor(agent_args.get('epsilon_end', 0.1)).float().to(self.device)
        self.epsilon_decay_param = torch.tensor(agent_args.get('epsilon_decay_param', 200000)).float().to(self.device)
        self.epsilon = self.epsilon_start

        # initializing the deep-learning parameters
        assert 'layer_sizes' in agent_args, "layer_sizes needs to be specified in agent_args"
        self.layer_sizes = agent_args['layer_sizes']
        self.num_actions = self.layer_sizes[-1]
        self.q_net = build_fc_net_tanh(self.layer_sizes).to(self.device)
        # print(self.q_net.state_dict())
        self.load_model_from = agent_args.get('load_model_from', None)
        if self.load_model_from:
            self.q_net.load_state_dict(torch.load(self.load_model_from))
            print(f'Successfully loaded model from {self.load_model_from}')
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.net_sync_freq = agent_args.get('net_sync_freq', 256)
        self.net_sync_counter = 0
        self.param_update_freq = agent_args.get('param_update_freq', 32)
        self.step_size = agent_args.get('step_size', 1e-3)
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer_name = agent_args.get('optimizer', 'None')
        self._initialize_optimizer()
        self.batch_size = agent_args.get('batch_size', 32)
        self.buffer_size = agent_args.get('buffer_size', 50000)
        self.experience_buffer = deque(maxlen=self.buffer_size)

        # initializing the two hyperparameters of centered discounted algorithms
        self.eta = torch.tensor(agent_args.get('eta', 0.01)).float().to(self.device)
        self.gamma = torch.tensor(agent_args.get('gamma', 0.95)).float().to(self.device)

        # initializing the average-reward parameter
        self.avg_reward_init = agent_args.get('avg_reward_init', 0)
        self.avg_reward = torch.tensor(self.avg_reward_init, requires_grad=False, dtype=torch.float)

        # initializing the step-size parameter for the average-reward update
        self.beta_init = self.eta * self.step_size
        self.robust_to_initialization = agent_args.get('robust_to_initialization', False)
        self.beta_sequence = 'unbiased_trick' if self.robust_to_initialization else 'constant'
        self._initialize_avgrew_step_size()

        # for a Q-learning-style update vs a Sarsa-style update
        self.sarsa_update = agent_args.get('sarsa_update', False)

        # for logistics and checkpointing
        self.timestep = 0
        self.save_model_loc = agent_args['output_folder'] + 'models/'
        validate_output_folder(self.save_model_loc)
        self.last_obs = None
        self.last_action = None
        self.max_value_per_step = None

    def _initialize_optimizer(self):
        assert self.optimizer_name in ['SGD', 'Adam', 'RMSprop'], "optimizer needs to be SGD, Adam, or RMSprop"
        if self.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.q_net.parameters(), lr=self.step_size)
        elif self.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.step_size)
        elif self.optimizer_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params=self.q_net.parameters(),
                                                 lr=self.step_size, alpha=0.95, eps=0.01)
        else:
            raise ValueError("optimizer needs to be SGD, Adam, or RMSprop")

    def _initialize_avgrew_step_size(self):
        """Initializes the step size for the average-reward parameter."""
        if self.beta_sequence == 'unbiased_trick':
            self.o_b = self.beta_init if self.beta_init != 0 else 1
            self.beta = self.beta_init / self.o_b
        else:
            self.beta = self.beta_init

    def _update_step_size(self):
        """Updates the reward-rate step size per step."""
        if self.beta_sequence == 'unbiased_trick':
            self.o_b = self.o_b + self.beta_init * (1 - self.o_b)
            self.beta = self.beta_init / self.o_b

    def save_trained_model(self, filename_suffix='dqn'):
        """Saves the trained model to a file."""
        filename = self.save_model_loc + filename_suffix + '.pth'
        torch.save(self.q_net.state_dict(), filename)

    def _choose_action(self, states):
        """Takes a batch of states and returns the e-greedy action for each."""
        if states.shape[0] == 1:    # single state
            if torch.rand(1) < self.epsilon:
                action = torch.randint(0, self.num_actions, (states.shape[0],))
                self.max_value_per_step = None
            else:
                with torch.no_grad():
                    qs = self.q_net(states)
                self.max_value_per_step, action = torch.max(qs, dim=1)
            return action
        else:                       # batch of states
            random_actions = torch.randint(low=0, high=self.num_actions, size=(states.shape[0],))
            with torch.no_grad():
                qs = self.q_net(states)
                _, greedy_actions = torch.max(qs, dim=1)
            actions = torch.where(torch.rand(states.shape[0]) < self.epsilon, random_actions, greedy_actions)
            return actions

    def _get_bootstrapping_values(self, next_state_vec, actions=None):
        """Returns the action values for a batch of next states, for bootstrapping."""
        with torch.no_grad():
            q_next_s = self.target_net(next_state_vec)     # ToDo: check why this doesn't need to(device)
        if actions is not None:     # Sarsa-type target
            q_next_sa = q_next_s.gather(1, actions)
            return q_next_sa
        else:                       # Q-learning-type target
            q_next_sa, _ = q_next_s.max(dim=1)
            return q_next_sa.unsqueeze(1)

    def _add_to_buffer(self, experience):
        """Adds a single experience to the experience buffer."""
        s, a, r, sn = experience
        # s = torch.tensor(np.expand_dims(s, 0), device=self.device).float()
        a = torch.tensor([[a]], device=self.device).long()
        r = torch.tensor([[r]], device=self.device).float()
        # sn = torch.tensor(np.expand_dims(sn, 0), device=self.device).float()
        self.experience_buffer.append([s, a, r, sn])

    def _sample_from_buffer(self):
        """Samples a batch of experiences from the experience buffer."""
        num_samples = min(self.buffer_size, len(self.experience_buffer))
        sample = random.sample(self.experience_buffer, num_samples)
        s, a, r, sn = zip(*sample)
        states = torch.cat(s, dim=0)
        actions = torch.cat(a, dim=0)
        rewards = torch.cat(r, dim=0)
        next_states = torch.cat(sn, dim=0)

        return states, actions, rewards, next_states

    def _update_epsilon(self):
        """Updates the epsilon parameter per step."""
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.e ** (
                        -self.timestep / self.epsilon_decay_param)

    def _process_raw_observation(self, obs: np.ndarray):
        """Takes an ndarray, flattens it, and returns it in a batch form.
        Args:
            obs: ndarray of arbitrary size
        Returns:
            tensor of shape (1, flattened_size_of_obs)
        """
        return torch.tensor(obs, dtype=torch.float, device=self.device).flatten().unsqueeze(0)

    def start(self, first_state):
        """Returns the first action corresponding to the first state."""
        observation = self._process_raw_observation(first_state)
        action = self._choose_action(observation)
        self.last_obs = observation
        self.last_action = action
        return action.item()    # return the actual integer instead of the 1D tensor

    def step(self, reward, next_state):
        """Updates the parameters and returns a new action."""
        self.timestep += 1

        observation = self._process_raw_observation(next_state)
        self._add_to_buffer([self.last_obs, self.last_action, reward, observation])

        # if time to update parameters
        if self.timestep % self.param_update_freq == 0:
            # if time to update target network
            if self.timestep % self.net_sync_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            # update the average-reward and q_net parameters
            self._update_params()
            # update exploration parameter
            self._update_epsilon()
            # update the step size
            self._update_step_size()

        action = self._choose_action(observation)
        self.last_obs = observation
        self.last_action = action
        return action.item()    # return the actual integer instead of the 1D tensor

    def _update_params(self):
        """Updates the parameters of the agent."""

        # sample a batch of transitions
        states, actions, rewards, next_states = self._sample_from_buffer()

        # predict expected return of current state using main network
        qs = self.q_net(states)
        pred_return = qs.gather(1, actions)

        # get target return using target network
        next_actions = None
        if self.sarsa_update:
            next_actions = self._choose_action(next_states).unsqueeze(1)
        q_next = self._get_bootstrapping_values(next_states, next_actions)
        with torch.no_grad():  # otherwise torch tries to backprop through self.avg_reward
            target_return = rewards - self.avg_reward + self.gamma * q_next

            # update the average-reward parameter
            old_avg_reward = self.avg_reward
            delta = (target_return - pred_return) if not self.sarsa_update else (rewards - self.avg_reward)
            self.avg_reward += self.beta * torch.mean(delta)

            # in case the new avg-rew parameter should be used right away
            if self.robust_to_initialization:
                target_return += (old_avg_reward - self.avg_reward)

        # update the q_net parameters
        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CDQNAgent(DeepCDiscAgent):
    """
    Implements DQN with reward centering (Naik, Wan, Tomar, Sutton, 2024).
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.sarsa_update = False


class CDSNAgent(DeepCDiscAgent):
    """
    Implements the Sarsa version of CDQN, 
    that is, the on-policy version, which uses a Sarsa-style update 
    and the on-policy update for the average-reward estimate.
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.sarsa_update = True
