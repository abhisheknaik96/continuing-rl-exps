import unittest
import numpy as np
from prediction_agents import DifferentialTDlambdaAgent


class TestDTDlAgent(unittest.TestCase):

    def test_get_representation(self):
        agent = DifferentialTDlambdaAgent(num_actions=3,
                                          num_features=2,
                                          pi=[0.7, 0.3])
        expected_rep_v = agent.get_representation([1, -1], -1)
        actual_rep_v = [1, -1]
        expected_rep_q = agent.get_representation([1, -1], 1).tolist()
        actual_rep_q = [0, 0, 1, -1, 0, 0]

        self.assertListEqual(expected_rep_v, actual_rep_v)
        self.assertListEqual(expected_rep_q, actual_rep_q)

    def test_get_value(self):
        agent = DifferentialTDlambdaAgent(num_actions=2,
                                          num_features=4,
                                          pi=[0.7, 0.3],
                                          value_init=1)
        dummy_states = np.array([[0, 1, 1, 0],
                                 [0, 1, 0, -1],
                                 [-1, -1, -1, -1],
                                 [0, 0, 0, 0],
                                 [100, -50, 0, 0]])
        expected_values = [2, 0, -4, 0, 50]
        values = agent.get_value(dummy_states).tolist()
        self.assertListEqual(values, expected_values)

    def test_choose_action(self):
        agent = DifferentialTDlambdaAgent(num_actions=2,
                                          num_features=1,
                                          pi=[0.7, 0.3])
        dummy_state = np.ones(1)
        num_samples = 100000
        num_action_1 = sum(agent.choose_action(dummy_state) for _ in range(num_samples))
        self.assertAlmostEqual(num_action_1/num_samples, agent.pi[1], places=2)

    def test_update_step_sizes(self):
        agent = DifferentialTDlambdaAgent(num_actions=2,
                                          num_features=1,
                                          pi=[0.7, 0.3],
                                          alpha_sequence='1byN')
        dummy_state = np.ones(1)
        dummy_reward = 0
        num_steps = 100
        agent.start(dummy_state)
        for i in range(num_steps):
            agent.step(dummy_reward, dummy_state)
        current_step_size = agent.alpha
        expected_step_size = 1 / (num_steps + 1)
        self.assertEqual(current_step_size, expected_step_size)


if __name__ == '__main__':
    unittest.main()
