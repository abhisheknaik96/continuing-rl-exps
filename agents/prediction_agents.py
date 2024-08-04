"""This files contains the implementation of the prediction agents."""

import numpy as np
from agents.base_agents import LFAPredictionAgent


class DifferentialDiscountedTDlambdaAgent(LFAPredictionAgent):
    """
    Implements multi-step TD learning with reward centering
    """
    def __init__(self, **agent_args):
        super().__init__(**agent_args)
        self.gamma = agent_args.get('gamma', 0.95)
        self.eta = agent_args.get('eta', 0.1)
        self.avg_reward_init = agent_args.get('avg_reward_init', 0)
        self.avg_reward = self.avg_reward_init
        self.avg_cost_update = agent_args.get('avg_cost_update', False)
        self.lamda = agent_args.get('lambda', 0.8)  # lambda is a keyword
        self.trace = np.zeros_like(self.weights)
        self.trace_r_bar = 0
        self.dtdlf = agent_args.get('dtdlf', False)

    def _update_weights(self, reward, state):
        prediction = self._get_value(self.past_state)
        target = reward - self.avg_reward + self.gamma * self._get_value(state)
        delta = target - prediction
        delta_r_bar = (reward - self.avg_reward) if self.avg_cost_update else delta

        self.trace = self.past_rho * (self.lamda * self.trace + self.past_state)
        self.trace_r_bar = self.past_rho * (self.lamda * self.trace_r_bar + 1) if self.dtdlf else 1

        self.weights += self.alpha * delta * self.trace
        self.avg_reward += self.eta * self.alpha * delta_r_bar * self.trace_r_bar
