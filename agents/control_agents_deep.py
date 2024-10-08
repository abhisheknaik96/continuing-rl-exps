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


class DeepBaseAgent:
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

        # initializing the deep-learning parameters
        self.net_sync_freq = agent_args.get('net_sync_freq', 256)
        self.net_sync_counter = 0
        self.param_update_freq = agent_args.get('param_update_freq', 32)
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
            # update target network
            self._update_target_net()
            # update the learnable parameters
            self._update_params()
            # update exploration parameters
            self._update_exploration_parameters()
            # update the step size
            self._update_step_size()

        action = self._choose_action(observation)
        self.last_obs = observation
        self.last_action = action
        return action

    def _choose_action(self, states):
        return NotImplementedError

    def _update_target_net(self):
        return NotImplementedError
    
    def _update_exploration_parameters(self):
        return NotImplementedError


class DeepCenteredDiscountedValueBasedAgent(DeepBaseAgent):
    """
    Implements the general form of discounted value-based algorithms with reward centering (Naik, Wan, Tomar, Sutton, 2024).
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        
        assert 'layer_sizes' in agent_args, "layer_sizes needs to be specified in agent_args"
        self.layer_sizes = agent_args['layer_sizes']
        self.num_actions = self.layer_sizes[-1]

        # initializing the action-value network and the corresponding target network
        self.q_net = build_fc_net_tanh(self.layer_sizes).to(self.device)
        self.load_model_from = agent_args.get('load_model_from', None)
        if self.load_model_from:
            self.q_net.load_state_dict(torch.load(self.load_model_from))
            print(f'Successfully loaded model from {self.load_model_from}')
        self.target_net = copy.deepcopy(self.q_net).to(self.device)

        # initializing the loss function and optimizer
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer_name = agent_args.get('optimizer', 'None')
        self._initialize_optimizer()
        self.step_size = agent_args.get('step_size', 1e-3)
        
        # initializing the parameters for epsilon-greedy action selection
        self.epsilon_start = torch.tensor(agent_args.get('epsilon_start', 0.9)).float().to(self.device)
        self.epsilon_end = torch.tensor(agent_args.get('epsilon_end', 0.1)).float().to(self.device)
        self.epsilon_decay_param = torch.tensor(agent_args.get('epsilon_decay_param', 200000)).float().to(self.device)
        self.epsilon = self.epsilon_start

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

    def _update_target_net(self):
        if self.timestep % self.net_sync_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _update_exploration_parameters(self):
        """Updates the epsilon parameter per step."""
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.e ** (
                        -self.timestep / self.epsilon_decay_param)

    def step(self, reward, next_state):
        action_tensor = super().step(reward, next_state)
        return action_tensor.item()    # return the actual integer instead of the 1D tensor


class CDQNAgent(DeepCenteredDiscountedValueBasedAgent):
    """
    Implements DQN with reward centering.
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.sarsa_update = False


class CDSNAgent(DeepCenteredDiscountedValueBasedAgent):
    """
    Implements the Sarsa version of CDQN, 
    that is, the on-policy version, which uses a Sarsa-style update 
    and the on-policy update for the average-reward estimate.
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.sarsa_update = True


class DeepCenteredDiscountedPolicyBasedAgent(DeepBaseAgent):
    """
    Implements the general form of discounted policy-based algorithms with reward centering.
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)

        # initialize the actor network (and its target network)
        assert 'actor_arch' in agent_args, "actor_arch needs to be specified in agent_args"
        self.actor_arch = agent_args['actor_arch']
        self.num_actions = self.actor_arch[-1]
        self.actor = build_fc_net_tanh(self.actor_arch).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        # initilize the critic network (and its target network)
        assert 'critic_arch' in agent_args, "critic_arch needs to be specified in agent_args"
        self.critic_arch = agent_args['critic_arch']
        assert self.critic_arch[0] == self.actor_arch[0] + self.actor_arch[-1], "the input of the critic network should be the concatenation of the state features and actions"
        assert self.critic_arch[-1] == 1, "the output of the critic network should be a single value"
        self.critic = build_fc_net_tanh(self.critic_arch).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.tau = agent_args.get('tau', 0.995) # parameter for target networks' soft updates

        # initialize the loss functions and optimizers
        self.actor_loss_fn = torch.nn.MSELoss()
        self.actor_optimizer_name = agent_args.get('actor_optimizer', 'None')
        self.actor_step_size = agent_args.get('actor_step_size', 1e-3)
        self._initialize_optimizer(self.actor, self.actor_optimizer_name, self.actor_step_size)
        self.critic_optimizer_name = agent_args.get('critic_optimizer', 'None')
        self.actor_step_size = agent_args.get('actor_step_size', 1e-3)
        self._initialize_optimizer(self.critic, self.critic_optimizer_name, self.critic_step_size)

    def _initialize_optimizer(network, optimizer_name, step_size):
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=network.parameters(), lr=step_size)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=network.parameters(), lr=step_size)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(params=network.parameters(), lr=step_size, alpha=0.95, eps=0.01)
        else:
            raise ValueError("optimizer needs to be SGD, Adam, or RMSprop")
        
    def _choose_action(self, states):
        """Takes a batch of states and returns the action for each."""
        with torch.no_grad():
            actions = self.actor(states)
        # SpinningUp documentation says Gaussian noise is good enough, don't need OU.
        noisy_actions = actions + torch.randn_like(actions)
        return noisy_actions

    def _update_params(self):
        """Updates the actor and critic parameters of the agent."""
        
        # sample a batch of transitions
        states, actions, rewards, next_states = self._sample_from_buffer()
        
        ### first, update the critic network
        q_current = self.critic(torch.cat([states, actions], dim=1))
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_return = rewards - self.avg_reward + self.gamma * q_next

            # update the average-reward parameter
            old_avg_reward = self.avg_reward
            delta = target_return - q_current
            self.avg_reward += self.beta * torch.mean(delta)

            # in case the new avg-rew parameter should be used right away
            if self.robust_to_initialization:
                target_return += (old_avg_reward - self.avg_reward)
        
        # update the q_net parameters
        critic_loss = self.critic_loss_fn(q_current, target_return)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        ### now, update the actor network
        actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actions], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def _update_target_net(self):
        for main_net, target_net in [(self.actor, self.actor_target), (self.critic, self.critic_target)]:
            for main, target in zip(main_net.parameters(), target_net.parameters()):
                target.data.mul_(self.tau)
                target.data.add_((1-self.tau) * main.data)

    def _update_exploration_parameters(self):
        # nothing to update here
        return
