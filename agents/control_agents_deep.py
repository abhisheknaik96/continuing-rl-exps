"""Implements the deep versions of the control agents."""

import warnings
import math
import random
import copy
from collections import deque
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from utils.helpers import validate_output_folder, load_obs_mean_and_std
from utils.ou_noise import OU_Noise


def build_fc_net(layer_sizes, activation=torch.nn.Tanh(), final_activation_layer=torch.nn.Identity(), ortho_init=False):
    """Returns a full-connected network with an argument-specified activation function."""
    assert len(layer_sizes) > 1
    layers = []
    for index in range(len(layer_sizes) - 1):
        linear = torch.nn.Linear(layer_sizes[index], layer_sizes[index + 1])
        if ortho_init:
            torch.nn.init.orthogonal_(linear.weight, 1)
            torch.nn.init.constant_(linear.bias, 0)
        act = activation if index < len(layer_sizes) - 2 else final_activation_layer
        layers += (linear, act)
    return torch.nn.Sequential(*layers)


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
        self.eval_mode = agent_args.get('eval_mode', False)

        # initializing the RNG seed for reproducibility
        self.seed = agent_args.get('rng_seed', 42)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(mode=True)
        random.seed(self.seed)  # ToDo: don't like this. Remove need for this package

        # initializing the deep-learning parameters
        self.net_sync_freq = agent_args.get('net_sync_freq', 256)
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
        self.step_size = agent_args.get('step_size', 1e-3) if not 'critic_step_size' in agent_args else agent_args['critic_step_size']
        self.beta_init = self.eta * self.step_size
        self.robust_to_initialization = agent_args.get('robust_to_initialization', False)
        self.beta_sequence = 'unbiased_trick' if self.robust_to_initialization else 'constant'
        self._initialize_avgrew_step_size()

        # for a Q-learning-style update vs a Sarsa-style update
        self.sarsa_update = agent_args.get('sarsa_update', False)

        # for normalizing observations
        assert 'num_features' in agent_args, "num_features needs to be specified in agent_args"
        self.num_features = agent_args['num_features']
        self.obs_mean = np.zeros(self.num_features)
        self.obs_m2 = np.ones(self.num_features) * 0.01
        self.obs_std = np.zeros(self.num_features)
        if self.eval_mode:
            self.load_obs_mean_std_from = agent_args['load_obs_mean_std_from']
            self.load_run_idx = agent_args['load_run_idx']
            self.obs_mean, self.obs_std = load_obs_mean_and_std(self.load_obs_mean_std_from, self.load_run_idx)
            print(f'Loaded observation mean and stddev for run {self.load_run_idx} from {self.load_obs_mean_std_from}.')
            # print(f'{self.obs_mean}\n{self.obs_std}\n')

        # for logistics and checkpointing
        self.timestep = 0
        self.save_model_loc = agent_args['output_folder'] + 'models/'
        validate_output_folder(self.save_model_loc)
        self.last_obs = None
        self.last_action = None
        self.max_value_per_step = None

    def _initialize_optimizer(self, network, optimizer_name, step_size):
        assert optimizer_name in ['SGD', 'Adam', 'RMSprop'], "optimizer needs to be SGD, Adam, or RMSprop"
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params=network.parameters(), lr=step_size)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params=network.parameters(), lr=step_size)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(params=network.parameters(), lr=step_size, alpha=0.95, eps=0.01)
        else:
            raise ValueError("optimizer needs to be SGD, Adam, or RMSprop")
        return optimizer

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
        
    def _add_to_buffer(self, experience):
        """Adds a single experience to the experience buffer."""
        s, a, r, sn = experience
        r = torch.tensor([[r]], device=self.device).float()
        self.experience_buffer.append([s, a, r, sn])

    def _sample_from_buffer(self, in_order=False):
        """Samples a batch of experiences from the experience buffer."""
        num_samples = min(self.batch_size, len(self.experience_buffer))
        if not in_order:
            sample = random.sample(self.experience_buffer, num_samples)
        else:
            raise NotImplementedError
        s, a, r, sn = zip(*sample)
        states = torch.cat(s, dim=0)
        actions = torch.cat(a, dim=0)
        rewards = torch.cat(r, dim=0)
        next_states = torch.cat(sn, dim=0)

        return states, actions, rewards, next_states

    def _process_raw_observation(self, obs: np.ndarray):
        """
        Takes an ndarray, normalizes it (using Welford's online algorithm), 
        flattens it, and returns it in a batch form.

        Args:
            obs: ndarray of arbitrary size
        Returns:
            tensor of shape (1, flattened_size_of_obs)
        """
        if not self.eval_mode:
            # update the running estimate of the mean
            delta = obs - self.obs_mean
            self.obs_mean += delta / self.timestep    

            # update the running estimate of the variance
            delta_2 = obs - self.obs_mean
            self.obs_m2 += delta * delta_2
            self.obs_std = np.sqrt(self.obs_m2 / self.timestep)

            # compute the normalized observation
            obs = (obs - self.obs_mean) / self.obs_std if self.timestep > 100 else obs

        else:
            obs = (obs - self.obs_mean) / self.obs_std

        return torch.tensor(obs, dtype=torch.float, device=self.device).flatten().unsqueeze(0)

    def start(self, first_state):
        """Returns the first action corresponding to the first state."""
        self.timestep += 1
        observation = self._process_raw_observation(first_state)
        with torch.no_grad():
            action = self._choose_action(observation)
        self.last_obs = observation
        self.last_action = action
        return action

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
            # update the step size
            self._update_step_size()
        # update exploration parameters
        self._update_exploration_parameters()

        with torch.no_grad():
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

    def save_trained_model(self):
        """Saves the trained model to a file."""
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
        self.q_net = build_fc_net(self.layer_sizes, activation=torch.nn.Tanh()).to(self.device)
        self.load_model_from = agent_args.get('load_model_from', None)
        if self.load_model_from:
            self.q_net.load_state_dict(torch.load(self.load_model_from))
            print(f'Successfully loaded model from {self.load_model_from}')
        self.target_net = copy.deepcopy(self.q_net).to(self.device)

        # initializing the loss function and optimizer
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SmoothL1Loss()
        self.optimizer_name = agent_args.get('optimizer', 'None')
        self.step_size = agent_args.get('step_size', 1e-3)
        self.optimizer = self._initialize_optimizer(self.q_net, self.optimizer_name, self.step_size)
        
        # initializing the parameters for epsilon-greedy action selection
        self.epsilon_start = torch.tensor(agent_args.get('epsilon_start', 0.9)).float().to(self.device)
        self.epsilon_end = torch.tensor(agent_args.get('epsilon_end', 0.1)).float().to(self.device)
        self.epsilon_decay_param = torch.tensor(agent_args.get('epsilon_decay_param', 200000)).float().to(self.device)
        self.epsilon = self.epsilon_start
        

    def _choose_action(self, states):
        """Takes a batch of states and returns the e-greedy action for each."""
        random_actions = torch.randint(low=0, high=self.num_actions, size=(states.shape[0],))
        with torch.no_grad():
            qs = self.q_net(states)
            _, greedy_actions = torch.max(qs, dim=1)
        actions = torch.where(torch.rand(states.shape[0]) < self.epsilon, random_actions, greedy_actions)
        return actions.unsqueeze(1)
        
    def _get_bootstrapping_values(self, next_state_vec, actions=None):
        """Returns the action values for a batch of next states, for bootstrapping."""
        with torch.no_grad():
            q_next_s = self.target_net(next_state_vec)
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
        pred_return = self.q_net(states)

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

    def start(self, first_state):
        action_tensor = super().start(first_state)
        return action_tensor.item()    # return the actual integer instead of the 1D tensor

    def step(self, reward, next_state):
        action_tensor = super().step(reward, next_state)
        return action_tensor.item()    # return the actual integer instead of the 1D tensor

    def save_trained_model(self, filename_suffix='dqn'):
        """Saves the trained model to a file."""
        filename = self.save_model_loc + filename_suffix + '.pth'
        torch.save(self.q_net.state_dict(), filename)


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
        self.ortho_init = agent_args.get('orthogonal_initialization', False)
        self.actor = build_fc_net(self.actor_arch, activation=torch.nn.ReLU(), final_activation_layer=torch.nn.Tanh(), 
                                  ortho_init=self.ortho_init).to(self.device)
        self.load_model_from = agent_args.get('load_model_from', None)
        if self.load_model_from is not None:
            self.actor.load_state_dict(torch.load(self.load_model_from, weights_only=True))
            print(f'Successfully loaded model from {self.load_model_from}')

        # initialize the critic network (and its target network)
        assert 'critic_arch' in agent_args, "critic_arch needs to be specified in agent_args"
        self.critic_arch = agent_args['critic_arch']
        assert self.critic_arch[-1] == 1, "the output of the critic network should be a single value"
        self.critic = build_fc_net(self.critic_arch, activation=torch.nn.ReLU(), ortho_init=self.ortho_init).to(self.device)

        # initialize the target networks, if any
        self.target_nets = agent_args.get('target_nets', True)
        if self.target_nets:
            self.actor_target = copy.deepcopy(self.actor).to(self.device)
            self.critic_target = copy.deepcopy(self.critic).to(self.device)
            self.tau = agent_args.get('tau', 0.995)     # parameter for target networks' soft updates
            self.net_target_pairs = [(self.actor, self.actor_target), (self.critic, self.critic_target)]

        # initialize the loss functions and optimizers
        self.actor_optimizer_name = agent_args.get('actor_optimizer', 'None')
        self.actor_step_size = agent_args.get('actor_step_size', 3e-4)
        self.actor_optimizer = self._initialize_optimizer(self.actor, self.actor_optimizer_name, self.actor_step_size)
        self.critic_loss_fn = torch.nn.MSELoss()
        self.critic_optimizer_name = agent_args.get('critic_optimizer', 'None')
        self.critic_step_size = agent_args.get('critic_step_size', 1e-3)
        self.critic_optimizer = self._initialize_optimizer(self.critic, self.critic_optimizer_name, self.critic_step_size)

        # initialize the exploration parameters
        self.initial_exploration_only_steps = agent_args.get('initial_exploration_only_steps', 5000)
        self.exploration_sigma_init = agent_args.get('exploration_sigma_init', 1)
        self.exploration_sigma_final = agent_args.get('exploration_sigma_final', 0.1)
        self.exploration_decay_type = agent_args.get('exploration_decay_type', 'linear')
        self.exploration_decay_param = agent_args.get('exploration_decay_param', 20000)
        self.exploration_sigma = self.exploration_sigma_init
        assert "num_max_steps" in agent_args, "num_max_steps needs to be specified in agent_args"
        self.num_max_steps = agent_args['num_max_steps']
    
    def _update_target_net(self):
        "Update the target networks with 'soft' updates."
        if self.target_nets and (self.timestep % self.net_sync_freq == 0):
            for main_net, target_net in self.net_target_pairs:
                for main, target in zip(main_net.parameters(), target_net.parameters()):
                    target.data.mul_(self.tau)          # these are in-place operations
                    target.data.add_((1-self.tau) * main.data)

    def _update_exploration_parameters(self):
        """Update the exploration parameter."""
        if self.timestep < self.initial_exploration_only_steps:
            return
        if self.timestep > 0.9 * self.num_max_steps:
            self.exploration_sigma = self.exploration_sigma_final
        if self.exploration_decay_type == 'linear':
            self.exploration_sigma -= (self.exploration_sigma_init - 
                                       self.exploration_sigma_final) / (0.9*self.num_max_steps - self.initial_exploration_only_steps)
        else:
            raise ValueError("only 'linear' exploration_decay_type supported at the moment")

    def save_trained_model(self, filename_suffix='DDPG'):
        """Saves the trained model to a file."""
        actor_filename = self.save_model_loc + filename_suffix + '_actor.pth'
        critic_filename = self.save_model_loc + filename_suffix + '_critic.pth'
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.critic.state_dict(), critic_filename)

    def start(self, first_state):
        action_tensor = super().start(first_state)
        return action_tensor[0].cpu().numpy()    # return the action array instead of the 2D tensor containing the single action array

    def step(self, reward, next_state):
        action_tensor = super().step(reward, next_state)
        return action_tensor[0].cpu().numpy()    # return the action array instead of the 2D tensor containing the single action array

    def _choose_action(self, states):
        """Takes a batch of states and returns the action for each."""
        raise NotImplementedError

    def _update_params(self):
        """Updates the actor and critic parameters of the agent."""
        raise NotImplementedError


class DDPGAgent(DeepCenteredDiscountedPolicyBasedAgent):
    """Implements the DDPG algorithm (Silver et al., 2016) with reward centering (Naik et al., 2024)."""

    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        assert self.critic_arch[0] == self.actor_arch[0] + self.actor_arch[-1], \
            "the input to the action-value critic network should be the concatenation of the state features and actions"
        self.target_nets = True
        self.use_ou_noise = agent_args.get('use_ou_noise', False)
        if self.use_ou_noise:
            self.ou_noise = OU_Noise(size=(1, self.num_actions), seed=self.seed)

    def _choose_action(self, states):
        """Takes a batch of states and returns the action for each."""
        if self.timestep < self.initial_exploration_only_steps:
            # noisy_actions = torch.rand((self.num_actions, 1)) * 2 - 1      # random action in [-1, 1]
            if self.use_ou_noise:
                noisy_actions = torch.from_numpy(self.ou_noise.sample())
            else:
                noisy_actions = torch.randint(-1, 2, (1, self.num_actions), dtype=torch.float32)     # random action in {-1, 0, 1}
        else:
            with torch.no_grad():
                actions = self.actor(states)
            # SpinningUp documentation says Gaussian noise is good enough, don't need OU.
            if self.use_ou_noise:
                noisy_actions = actions + self.ou_noise.sample()
            else:
                noisy_actions = torch.normal(actions, self.exploration_sigma)
        noisy_actions = torch.clip(noisy_actions, -1, 1)

        return noisy_actions.to(dtype=torch.float32)

    def _update_params(self):
        """Updates the actor and critic parameters of the agent."""
        
        if self.timestep < self.initial_exploration_only_steps:
            return

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
        
        # the critic params won't be changed during the actor update, so disable any gradient computation for them
        for p in self.critic.parameters():
            p.requires_grad = False

        ### now, update the actor network
        actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actions], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # enable the gradient computation for the critic parameters for the next call to _update_params()
        for p in self.critic.parameters():
            p.requires_grad = True


class TD3Agent(DeepCenteredDiscountedPolicyBasedAgent):
    """Implements the TD3 algorithm (Fujimoto et al., 2018) with reward centering (Naik et al., 2024)."""

    def __init__(self, **agent_args):
    
        super(DeepCenteredDiscountedPolicyBasedAgent, self).__init__(**agent_args)

        # initialize the actor network
        assert 'actor_arch' in agent_args, "actor_arch needs to be specified in agent_args"
        self.actor_arch = agent_args['actor_arch']
        self.num_actions = self.actor_arch[-1]
        self.ortho_init = agent_args.get('orthogonal_initialization', False)
        self.actor = build_fc_net(self.actor_arch, activation=torch.nn.ReLU(), final_activation_layer=torch.nn.Tanh(), 
                                  ortho_init=self.ortho_init).to(self.device)
        self.load_model_from = agent_args.get('load_model_from', None)
        if self.load_model_from is not None:
            self.actor.load_state_dict(torch.load(self.load_model_from, weights_only=True))
            print(f'Successfully loaded model from {self.load_model_from}')

        # initialize the critic networks
        assert 'critic_arch' in agent_args, "critic_arch needs to be specified in agent_args"
        self.critic_arch = agent_args['critic_arch']
        assert self.critic_arch[0] == self.actor_arch[0] + self.num_actions, \
            "the input to the action-value critic network should be the concatenation of the state features and actions"
        self.critic_0 = build_fc_net(self.critic_arch, activation=torch.nn.ReLU(), ortho_init=self.ortho_init).to(self.device)
        self.critic_1 = build_fc_net(self.critic_arch, activation=torch.nn.ReLU(), ortho_init=self.ortho_init).to(self.device)

        # initialize the target networks
        self.target_nets = True
        if self.target_nets:
            self.actor_target = copy.deepcopy(self.actor).to(self.device)
            self.critic_0_target = copy.deepcopy(self.critic_0).to(self.device)
            self.critic_1_target = copy.deepcopy(self.critic_1).to(self.device)
            self.tau = agent_args.get('tau', 0.995) # parameter for target networks' soft updates
        self.net_target_pairs = [(self.actor, self.actor_target), (self.critic_0, self.critic_0_target), (self.critic_1, self.critic_1_target)]

        # initialize the loss functions and optimizers
        self.actor_optimizer_name = agent_args.get('actor_optimizer', 'None')
        self.actor_step_size = agent_args.get('actor_step_size', 3e-4)
        self.actor_optimizer = self._initialize_optimizer(self.actor, self.actor_optimizer_name, self.actor_step_size)
        self.critic_optimizer_name = agent_args.get('critic_optimizer', 'None')
        self.critic_step_size = agent_args.get('critic_step_size', 1e-3)
        self.critic_loss_fn = torch.nn.MSELoss()
        # self.critic_optimizer = self._initialize_optimizer(self.critic_0, self.critic_optimizer_name, self.critic_step_size)
        self.critic_optimizer = torch.optim.Adam(params=list(self.critic_0.parameters()) + list(self.critic_1.parameters()), 
                                                 lr=self.critic_step_size)      # ToDo: add support for other optimizers
        warnings.warn("The current SAC implementation only supports the Adam optimizer for the critic networks.")

        # initialize the parameter for delaying the actor updates
        self.actor_delay_wrt_critic = agent_args.get('actor_delay_wrt_critic', 2)
        self.actor_update_counter = 0

        # initialize exploration and smoothing parameters
        self.noise_clip_param = agent_args.get('noise_clip_param', 0.5)
        assert self.noise_clip_param > 0, 'noise_clip_param should be positive'
        self.smoothing_sigma = agent_args.get('smoothing_sigma', 0.2)
        self.initial_exploration_only_steps = agent_args.get('initial_exploration_only_steps', 5000)
        self.exploration_sigma_init = agent_args.get('exploration_sigma_init', 1)
        self.exploration_sigma_final = agent_args.get('exploration_sigma_final', 0.1)
        self.exploration_decay_type = agent_args.get('exploration_decay_type', 'linear')
        self.exploration_decay_param = agent_args.get('exploration_decay_param', 20000)
        self.exploration_sigma = self.exploration_sigma_init
        assert "num_max_steps" in agent_args, "num_max_steps needs to be specified in agent_args"
        self.num_max_steps = agent_args['num_max_steps']

    def _choose_action(self, states, for_target=False):
        """Takes a batch of states and returns the action for each."""
        if self.timestep < self.initial_exploration_only_steps:
            return torch.rand((1, self.num_actions)) * 2 - 1      # random actions in [-1, 1]
        
        with torch.no_grad():
            actions = self.actor(states)    
        noisy_actions = torch.normal(actions, self.exploration_sigma).clip(-1.0, 1.0)
        
        return noisy_actions
    
    def _update_params(self):
        """Updates the actor and critic parameters of the agent."""
        
        if self.timestep < self.initial_exploration_only_steps:
            return

        # sample a batch of transitions
        states, actions, rewards, next_states = self._sample_from_buffer()
        
        ### first, update the critic network
        q_current_0 = self.critic_0(torch.cat([states, actions], dim=1))
        q_current_1 = self.critic_1(torch.cat([states, actions], dim=1))
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # add (clipped) smoothing noise to the target action
            noise = (torch.randn_like(next_actions) * self.smoothing_sigma).clip(-self.noise_clip_param, self.noise_clip_param)
            next_actions += noise.clip(-1.0, 1.0)
            q_next_0 = self.critic_0_target(torch.cat([next_states, next_actions], dim=1))
            q_next_1 = self.critic_1_target(torch.cat([next_states, next_actions], dim=1))
            q_next = torch.min(q_next_0, q_next_1)
            target_return = rewards - self.avg_reward + self.gamma * q_next
                    
            # update the average-reward parameter
            old_avg_reward = self.avg_reward
            delta = target_return - torch.min(q_current_0, q_current_1)
            self.avg_reward += self.beta * delta.mean()

            # in case the new avg-rew parameter should be used right away
            if self.robust_to_initialization:
                target_return += (old_avg_reward - self.avg_reward)

        critic_0_loss = self.critic_loss_fn(q_current_0, target_return)
        critic_1_loss = self.critic_loss_fn(q_current_1, target_return)
        critic_loss = critic_0_loss + critic_1_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ### now, update the actor network
        self.actor_update_counter = (self.actor_update_counter + 1) % self.actor_delay_wrt_critic 
        if self.actor_update_counter == 0:
            actions = self.actor(states)      # sample new actions for the current states
            q_current_0 = self.critic_0(torch.cat([states, actions], dim=1))
            q_current_1 = self.critic_1(torch.cat([states, actions], dim=1))
            q_current = torch.min(q_current_0, q_current_1)     # Note: this is different from the original TD3 paper.
            actor_loss = -q_current.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

    def save_trained_model(self, filename_suffix='TD3'):
        """Saves the trained model to a file."""
        actor_filename = self.save_model_loc + filename_suffix + '_actor.pth'
        critic_0_filename = self.save_model_loc + filename_suffix + '_critic0.pth'
        critic_1_filename = self.save_model_loc + filename_suffix + '_critic1.pth'
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.critic_0.state_dict(), critic_0_filename)
        torch.save(self.critic_1.state_dict(), critic_1_filename)


class SACAgent(DeepCenteredDiscountedPolicyBasedAgent):
    """Implements the SAC algorithm (Haarnoja et al., 2018) with reward centering (Naik et al., 2024)."""

    def __init__(self, **agent_args):
    
        super(DeepCenteredDiscountedPolicyBasedAgent, self).__init__(**agent_args)

        # initialize the actor network
        assert 'actor_arch' in agent_args, "actor_arch needs to be specified in agent_args"
        self.actor_arch = agent_args['actor_arch']
        assert self.actor_arch[-1] % 2 == 0, "the actor's last layer should be of size 2*num_actions"
        self.num_actions = self.actor_arch[-1] // 2
        self.ortho_init = agent_args.get('orthogonal_initialization', False)
        self.actor = build_fc_net(self.actor_arch, activation=torch.nn.ReLU(), 
                                  final_activation_layer=torch.nn.Identity(), ortho_init=self.ortho_init).to(self.device)
        self.load_model_from = agent_args.get('load_model_from', None)
        if self.load_model_from is not None:
            self.actor.load_state_dict(torch.load(self.load_model_from, weights_only=True))
            print(f'Successfully loaded model from {self.load_model_from}')
        self.logstddev_min = agent_args.get('logstddev_min', -20)
        self.logstddev_max = agent_args.get('logstddev_max', 2)

        # initialize the critic networks
        assert 'critic_arch' in agent_args, "critic_arch needs to be specified in agent_args"
        self.critic_arch = agent_args['critic_arch']
        assert self.critic_arch[0] == self.actor_arch[0] + self.num_actions, \
            "the input to the action-value critic network should be the concatenation of the state features and actions"
        self.critic_0 = build_fc_net(self.critic_arch, activation=torch.nn.ReLU(), ortho_init=self.ortho_init).to(self.device)
        self.critic_1 = build_fc_net(self.critic_arch, activation=torch.nn.ReLU(), ortho_init=self.ortho_init).to(self.device)

        # initialize the target networks
        self.target_nets = True
        if self.target_nets:
            self.critic_0_target = copy.deepcopy(self.critic_0).to(self.device)
            self.critic_1_target = copy.deepcopy(self.critic_1).to(self.device)
            self.tau = agent_args.get('tau', 0.995) # parameter for target networks' soft updates
        self.net_target_pairs = [(self.critic_0, self.critic_0_target), (self.critic_1, self.critic_1_target)]

        # initialize the loss functions and optimizers
        self.actor_optimizer_name = agent_args.get('actor_optimizer', 'None')
        self.actor_step_size = agent_args.get('actor_step_size', 3e-4)
        self.actor_optimizer = self._initialize_optimizer(self.actor, self.actor_optimizer_name, self.actor_step_size)
        self.critic_optimizer_name = agent_args.get('critic_optimizer', 'None')
        self.critic_step_size = agent_args.get('critic_step_size', 1e-3)
        self.critic_loss_fn = torch.nn.MSELoss()
        # self.critic_optimizer = self._initialize_optimizer(self.critic_0, self.critic_optimizer_name, self.critic_step_size)
        self.critic_optimizer = torch.optim.Adam(params=list(self.critic_0.parameters()) + list(self.critic_1.parameters()), 
                                                 lr=self.critic_step_size)      # ToDo: add support for other optimizers
        warnings.warn("The current SAC implementation only supports the Adam optimizer for the critic networks.")

        # initialize the exploration parameters
        self.initial_exploration_only_steps = agent_args.get('initial_exploration_only_steps', 5000)
        self.initial_entropy_coeff = agent_args.get('initial_entropy_coeff', 0.2)
        self.log_entropy_coeff = torch.tensor(np.log(self.initial_entropy_coeff), requires_grad=True, device=self.device)
        self.log_entropy_coeff_optimizer = torch.optim.Adam(params=[self.log_entropy_coeff], lr=self.critic_step_size)
        self.target_entropy = -self.num_actions

        assert "num_max_steps" in agent_args, "num_max_steps needs to be specified in agent_args"
        self.num_max_steps = agent_args['num_max_steps']

    def _update_exploration_parameters(self):
        return

    def _get_mean_stddev_from_actor(self, states):
        means_and_log_stddevs = self.actor(states)          # check if slicing is an issue
        means = means_and_log_stddevs[:,:self.num_actions]
        log_stddevs = means_and_log_stddevs[:,self.num_actions:]
        log_stddevs = torch.clamp(log_stddevs, self.logstddev_min, self.logstddev_max)      # ToDo: use a natural bound via a tanh or something?
        stddevs = log_stddevs.exp()
        return means, stddevs

    def _evaluate_policy(self, states, exploration=True, compute_log_probs=True):
        """Takes a batch of states and returns the action for each."""

        means, stddevs = self._get_mean_stddev_from_actor(states)
        action_distributions = MultivariateNormal(means, torch.diag_embed(stddevs)) # ToDo: can avoid creating this in case of non-exploratory actions
        
        # sampling with the reparameterization trick
        actions_pre_squashing = action_distributions.rsample() if exploration else means

        if compute_log_probs:
            # compute the log probability of the actions (with some additional computation to account for the squashing)
            action_log_probabilities = action_distributions.log_prob(actions_pre_squashing)
            action_log_probabilities -= (2*(np.log(2) - actions_pre_squashing - torch.nn.functional.softplus(-2 * actions_pre_squashing))).sum(axis=1)
        else:
            action_log_probabilities = None
        actions = torch.tanh(actions_pre_squashing)                   # squashing the actions to be in [-1, 1]

        return actions, action_log_probabilities

    def _choose_action(self, observation):
        if self.timestep < self.initial_exploration_only_steps:
            action = torch.rand((1, self.num_actions)) * 2 - 1      # random actions in [-1, 1]
        else:
            with torch.no_grad():
                action, _ = self._evaluate_policy(observation, compute_log_probs=False)
        return action

    def entropy_coeff(self):
        return self.log_entropy_coeff.detach().exp()

    def _toggle_critic_gradient_computation(self, status):
        for p in self.critic_0.parameters():
            p.requires_grad = status
        for p in self.critic_1.parameters():
            p.requires_grad = status

    def _update_params(self):
        if self.timestep < self.initial_exploration_only_steps:
            return

        # sample a batch of transitions
        states, actions, rewards, next_states = self._sample_from_buffer()
        
        ### first, update the critic network
        q_current_0 = self.critic_0(torch.cat([states, actions], dim=1))
        q_current_1 = self.critic_1(torch.cat([states, actions], dim=1))
        with torch.no_grad():
            next_actions, next_action_log_probabilities = self._evaluate_policy(next_states)
            q_next_0 = self.critic_0_target(torch.cat([next_states, next_actions], dim=1))
            q_next_1 = self.critic_1_target(torch.cat([next_states, next_actions], dim=1))
            q_next = torch.min(q_next_0, q_next_1)
            target_return = rewards - self.avg_reward + self.gamma * (q_next - self.entropy_coeff() * next_action_log_probabilities)
        
            # update the average-reward parameter
            old_avg_reward = self.avg_reward
            delta = target_return - torch.min(q_current_0, q_current_1)
            self.avg_reward += self.beta * delta.mean()

            # in case the new avg-rew parameter should be used right away
            if self.robust_to_initialization:
                target_return += (old_avg_reward - self.avg_reward)

        critic_0_loss = self.critic_loss_fn(q_current_0, target_return)
        critic_1_loss = self.critic_loss_fn(q_current_1, target_return)
        critic_loss = critic_0_loss + critic_1_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # the critic params won't be changed during the actor update, so disable any gradient computation for them
        # self._toggle_critic_gradient_computation(False)

        ### now, update the actor network
        actions, action_log_probabilities = self._evaluate_policy(states)      # sample new actions for the current states
        q_current_0 = self.critic_0(torch.cat([states, actions], dim=1))
        q_current_1 = self.critic_1(torch.cat([states, actions], dim=1))
        q_current = torch.min(q_current_0, q_current_1)
        actor_obj = q_current - self.entropy_coeff() * action_log_probabilities
        actor_loss = -actor_obj.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ### finally, update the (log) entropy coefficient
        entropy_loss = -self.log_entropy_coeff.exp() * (self.target_entropy + action_log_probabilities.detach().mean())
        self.log_entropy_coeff_optimizer.zero_grad()
        entropy_loss.backward()
        self.log_entropy_coeff_optimizer.step()

        # re-enable gradient computation for the critic networks
        # self._toggle_critic_gradient_computation(True)

    def save_trained_model(self, filename_suffix='SAC'):
        """Saves the trained model to a file."""
        actor_filename = self.save_model_loc + filename_suffix + '_actor.pth'
        critic_0_filename = self.save_model_loc + filename_suffix + '_critic0.pth'
        critic_1_filename = self.save_model_loc + filename_suffix + '_critic1.pth'
        torch.save(self.actor.state_dict(), actor_filename)
        torch.save(self.critic_0.state_dict(), critic_0_filename)
        torch.save(self.critic_1.state_dict(), critic_1_filename)


class PPOAgent(DeepCenteredDiscountedPolicyBasedAgent):
    """Implements the PPO algorithm (Schulman et al., 2017) with reward centering (Naik et al., 2024)."""

    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.target_nets = False
        self.num_epochs_per_update = agent_args.get('num_epochs_per_update', 10)
        self.buffer_size = self.param_update_freq
        self.experience_buffer = deque(maxlen=self.buffer_size)
        assert self.batch_size <= self.param_update_freq and self.param_update_freq % self.batch_size == 0, \
            "param_update_freq should be a multiple of batch_size"
        self.normalize_advantage = agent_args.get('normalize_advantage', False)
        self.obj_clip_epsilon = agent_args.get('obj_clip_epsilon', 0.2)
        self.entropy_weight = agent_args.get('entropy_weight', 0.00)
        self.gae_lambda = agent_args.get('gae_lambda', 0.95)

    def _choose_action(self, observation):
        with torch.no_grad():
            action, action_log_prob, _ = self._evaluate_policy(observation)
        return action, action_log_prob

    def start(self, first_obs):
        """Returns the first action corresponding to the first state."""
        self.timestep += 1
        observation = self._process_raw_observation(first_obs)
        action, action_log_prob = self._choose_action(observation) 
        
        self.last_obs = observation
        self.last_action = action
        self.last_action_log_prob = action_log_prob
        return torch.clip(action[0], -1, 1).cpu().numpy()

    def step(self, reward, next_state):
        """Updates the parameters and returns a new action."""

        observation = self._process_raw_observation(next_state)
        # print(f"{self.timestep}: ", self.last_obs, self.last_action, reward, observation, self.last_action_log_prob)
        self._add_to_buffer([self.last_obs, self.last_action, reward, observation, self.last_action_log_prob])

        # if time to update parameters
        if self.timestep % self.param_update_freq == 0:
            # update the target networks, if any
            self._update_target_net()
            # update the learnable parameters
            self._update_params()
            # update the step size(s)
            self._update_step_size()
        # update the exploration parameters
        self._update_exploration_parameters()
        self.timestep += 1
        
        action, action_log_prob = self._choose_action(observation)
        
        self.last_obs = observation
        self.last_action = action
        self.last_action_log_prob = action_log_prob
        return torch.clip(action[0], -1, 1).cpu().numpy()

    def _add_to_buffer(self, experience):
        """Adds a single experience to the experience buffer."""
        s, a, r, sn, action_log_prob = experience
        r = torch.tensor([[r]], device=self.device).float()
        # action_log_prob = torch.tensor([action_log_prob], device=self.device).float()
        self.experience_buffer.append([s, a, r, sn, action_log_prob])

    def _sample_from_buffer(self):
        """Samples a batch of experiences from the experience buffer."""
        sample = copy.copy(self.experience_buffer)

        s, a, r, sn, a_log_prob = zip(*sample)
        states = torch.cat(s, dim=0)
        actions = torch.cat(a, dim=0)
        rewards = torch.cat(r, dim=0)
        next_states = torch.cat(sn, dim=0)
        action_log_probs = torch.cat(a_log_prob, dim=0)

        return states, actions, rewards, next_states, action_log_probs

    def _evaluate_policy(self, states, noisy_actions=None):
        """
        Takes a batch of states and returns the action and its log_probability for each,
        along with the policy's entropy.
        """
        actions = self.actor(states)

        exploration_covariance_matrix = torch.eye(self.num_actions).unsqueeze(0) * self.exploration_sigma        # ToDo: can avoid recreating this each time
        action_distribution = MultivariateNormal(actions, exploration_covariance_matrix.repeat(actions.shape[0], 1, 1))

        if noisy_actions is None:
            noisy_actions = action_distribution.sample()
        action_log_probability = action_distribution.log_prob(noisy_actions)
        entropy = action_distribution.entropy()
        
        return noisy_actions.to(dtype=torch.float32), action_log_probability.unsqueeze(1), entropy

    def _compute_returns_advantages(self, rewards, states, next_states, trajectory_length):
        with torch.no_grad():
            v_current = self.critic(states)
            v_next = self.critic(next_states)

        td_errors = rewards - self.avg_reward + self.gamma * v_next - v_current

        # update the average-reward estimate
        old_avg_reward = self.avg_reward
        self.avg_reward += self.beta * td_errors.mean()
        if self.robust_to_initialization:                       # update the TD errors with the new average-reward estimate
            td_errors += (old_avg_reward - self.avg_reward) 

        # compute advantages as a sum of TD errors 
        advantages = torch.zeros_like((v_current), device=self.device)
        advantages[-1] = td_errors[-1]
        for i in range(0, trajectory_length-1)[::-1]:
            advantages[i] = td_errors[i] + self.gamma * self.gae_lambda * advantages[i+1]
        # use the relation: advantage(state) = return(state) - value(state)
        returns = advantages + v_current

        return returns, advantages

    def _update_params(self):
        """Updates the actor and critic parameters of the agent."""
        
        if self.timestep < self.initial_exploration_only_steps:
            return

        # sample a batch of transitions
        states_all, actions_all, rewards_all, next_states_all, action_log_probs_all = self._sample_from_buffer()

        # compute returns and advantages
        trajectory_length = rewards_all.shape[0]
        returns_all, advantages_all = self._compute_returns_advantages(rewards_all, states_all, next_states_all, trajectory_length)

        # normalize advantages
        if self.normalize_advantage:
            advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-5)
        
        for _ in range(self.num_epochs_per_update):

            # shuffle the indices for minibatch updates within the epochs
            indices = random.sample(list(range(trajectory_length)), k=trajectory_length)
            minibatch_indices = np.array_split(indices, trajectory_length // self.batch_size)

            for idx in minibatch_indices:
                states = states_all[idx]
                actions = actions_all[idx]
                returns = returns_all[idx]
                action_log_probs = action_log_probs_all[idx]
                advantages = advantages_all[idx]

                ### first, update the critic parameters
                v_current = self.critic(states)
                critic_loss = self.critic_loss_fn(v_current, returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                ### now, update the actor parameters
                _, action_log_probs_latest, entropy_latest = self._evaluate_policy(states, actions)
                ratios = torch.exp(action_log_probs_latest - action_log_probs)
                actor_objective_cpi_term1 = ratios * advantages
                actor_objective_cpi_term2 = torch.clip(ratios, 1 - self.obj_clip_epsilon, 1 + self.obj_clip_epsilon) * advantages
                actor_objective_cpi = -torch.min(actor_objective_cpi_term1, actor_objective_cpi_term2).mean()
                actor_loss = actor_objective_cpi - self.entropy_weight * entropy_latest.mean()      # negative sign to maximize entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


class MDPOAgent(PPOAgent):
    """Implements Tomar et al.'s (2021) MDPO algorithm."""

    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.kl_coeff = 1

    def _update_kl_coeff(self):
        """Updates the extent to which the KL term will be used in the loss."""
        self.kl_coeff = 1 - self.timestep / self.num_max_steps

    def _update_params(self):
        """Updates the actor and critic parameters of the agent."""
        
        if self.timestep < self.initial_exploration_only_steps:
            return

        # sample a batch of transitions
        states_all, actions_all, rewards_all, next_states_all, action_log_probs_all = self._sample_from_buffer()

        # compute returns and advantages
        trajectory_length = rewards_all.shape[0]
        returns_all, advantages_all = self._compute_returns_advantages(rewards_all, states_all, next_states_all, trajectory_length)

        # normalize advantages
        if self.normalize_advantage:
            advantages_all = (advantages_all - advantages_all.mean()) / (advantages_all.std() + 1e-5)
        
        # update the factor with which the KL term is used in the loss
        self._update_kl_coeff()

        for _ in range(self.num_epochs_per_update):

            # shuffle the indices for minibatch updates within the epochs
            indices = random.sample(list(range(trajectory_length)), k=trajectory_length)
            minibatch_indices = np.array_split(indices, trajectory_length // self.batch_size)

            for idx in minibatch_indices:
                states = states_all[idx]
                actions = actions_all[idx]
                returns = returns_all[idx]
                action_log_probs = action_log_probs_all[idx]
                advantages = advantages_all[idx]

                ### first, update the critic parameters
                v_current = self.critic(states)
                critic_loss = self.critic_loss_fn(v_current, returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                ### now, update the actor parameters
                _, action_log_probs_latest, _ = self._evaluate_policy(states, actions)
                log_ratios = action_log_probs_latest - action_log_probs
                ratios = torch.exp(log_ratios)
                actor_objective_cpi = ratios * advantages
                actor_objective_kl = ratios * log_ratios - (ratios - 1)
                actor_loss = -(actor_objective_cpi.mean() - self.kl_coeff * actor_objective_kl.mean())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
