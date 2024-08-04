"""Helper functions required in the project."""

import os
import json
import numpy as np


def validate_output_folder(path):
    """Checks if folder exists. If not, creates it and returns its name"""
    path = os.path.join(path, "")      # to ensure path ends with '/'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_weights_from_npy(filename, seed_idx=-1):
    """
    Returns the weights from an npy file storing [runs, weights].

    Args:
        filename: full path of npy file which stores the weights
        seed_idx: which seed's weights to return; -1 returns average over seeds
    Returns:
        weights: average of the weights stored in the file
    """
    data = np.load(filename, allow_pickle=True).item()
    assert isinstance(data, dict)

    if seed_idx == -1:
        weights = np.mean(data['weights_final'], axis=0)
    else:
        weights = data['weights_final'][seed_idx]
    return weights


def get_centered_values(env, config):
    with open("environments/centered_values.json") as f:
        centered_values_all = json.load(f)
    try:
        policy = str(config['pi'][0])
        centered_values = centered_values_all[env][policy]
        if env == "RandomWalkN":
            num_states = config['num_states']
            k = num_states // 2
            values = np.array([1/(k+1)*i for i in range(-k, k+1)])
            centered_values['d_pi'] = [1/(k+1)*i/(k+1) for i in range(1,k+1)] + [1/(k+1)] + [1/(k+1)*i/(k+1) for i in range(1,k+1)][::-1]
            r_pi = centered_values['r_pi']
            reward_scale_factor = config['reward_scale_factor'] if 'reward_scale_factor' in config else 1.0
            reward_offset = config['reward_offset'] if 'reward_offset' in config else 0.0
            centered_values['v_pi'] = values * reward_scale_factor + reward_offset
            centered_values['r_pi'] = r_pi * reward_scale_factor + reward_offset
    except Exception as e:
        print(e)
        print("Something went wrong. Have the centered_values for this policy in this environment not saved"
              "in environments/centered_values.json?")
        raise

    return centered_values


def compute_rmsve(features, weights, targets, weighting, remove_offset=False, bias=False):
    """
    Computes the (weighted) root mean squared value error (RMSVE) of the
    estimated linear values w.r.t. given targets.
    There is an option to mean-center the estimates before the RMSVE is
    computed, as per Wan, Naik, Sutton's Appendix C.4 (2021).

    Args:
        features: the feature matrix X
        weights: the learned weights
        targets: the true values
        weighting: the distribution over states
        remove_offset: (bool) if the estimates should be mean-centered or not
    """
    if bias:
        features = np.hstack((features, np.ones((features.shape[0], 1))))
    estimates = np.dot(features, weights)
    if remove_offset:
        estimates -= np.dot(weighting, estimates)
    rmsve = np.sqrt(np.dot((targets - estimates) ** 2, weighting))
    return rmsve


def compute_rre(reward_rate_estimate, true_reward_rate):
    """Computes and returns the reward-rate error (RRE)."""
    return abs(reward_rate_estimate - true_reward_rate)
