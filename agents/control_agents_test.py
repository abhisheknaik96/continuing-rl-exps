import unittest
import numpy as np
# from agents.control_agents import DifferentialQlearningAgent


class TestDiffQAgent(unittest.TestCase):

    def test_get_representation(self):
        agent = DifferentialQlearningAgent(num_actions=3,
                                           num_features=2)
        expected_rep = [0, 0, 1, -1, 0, 0]
        obtained_rep = agent.get_representation([1, -1], 1).tolist()

        self.assertListEqual(expected_rep, obtained_rep)

    def test_argmax(self):
        num_actions = 3
        epsilon = 0.3
        agent = DifferentialQlearningAgent(num_actions=num_actions,
                                           num_features=1,
                                           epsilon=epsilon)
        dummy_obs = np.array([1])
        weights = np.array([1, 2, 3])
        agent._set_weights(weights)
        expected_action_distribution = [0.1, 0.1, 0.8]

        num_test_steps = 10000
        action_histogram = np.zeros(num_actions)
        for i in range(num_test_steps):
            action = agent.choose_action_egreedy(dummy_obs)
            action_histogram[action] += 1
        obtained_action_distribution = action_histogram / num_test_steps

        np.testing.assert_array_almost_equal(expected_action_distribution,
                                             obtained_action_distribution,
                                             decimal=2)


if __name__ == '__main__':
    # unittest.main()

    test_data = np.arange(10, dtype=np.float32)[..., np.newaxis]
    test_data = np.concatenate((test_data, -test_data), axis=1)
    # test_data = np.concatenate((test_data, test_data[::-1]), axis=1)

    current_obs_mean = np.array([0.0, 0.0])
    current_obs_m2 = np.array([0.001, 0.001])
    n = 0

    for i in range(test_data.shape[0]):
        obs = test_data[i]
        n += 1
        
        # update the running estimate of the mean
        delta = obs - current_obs_mean
        current_obs_mean += delta / n
        
        # update the running estimate of the variance
        delta_2 = obs - current_obs_mean
        current_obs_m2 += delta * delta_2
        obs_std = np.sqrt(current_obs_m2 / n)

        print(obs, current_obs_mean, obs_std, (obs - current_obs_mean) / obs_std)
