"""This files runs an experiment for a particular parameter configuration."""

import time
import sys
import os
import zipfile
import glob
from tqdm import tqdm
import numpy as np
from gym.envs.classic_control import rendering
import csuite
from utils import helpers
from environments import *
from agents.prediction_agents import *
from agents.control_agents import *
from agents.control_agents_deep import *

env_map = {'RandomWalkN': 'RandomWalkN',
           'TwoChoice': 'TwoChoiceMDP',
           'Loop': 'LoopN',
           'taxi': 'taxi',
           'catch': 'catch',
           'AC': 'access_control',
           'pendulum': 'pendulum',
           'puckworld': 'puckworld',
           'acrobot': 'Acrobot',
           'RW': 'RandomWalkN',
           'bandit': 'MultiArmedBandit',
           'RiverSwim': 'RiverSwim',
           }
agent_map = {'DTDl': 'DifferentialTDlambdaAgent',
             'ATDl': 'AverageCostTDlambdaAgent',
             'DiffDiscTD': 'DifferentialDiscountedTDlearningAgent',
             'DiffDiscTDl': 'DifferentialDiscountedTDlambdaAgent',
             'DiffDiscQ': 'DifferentialDiscountedQlearningAgent',
             'DiffDiscSarsa': 'DifferentialDiscountedSarsaAgent',
             'DiffQN': 'DiffQNAgent',
             'DQN': 'DQNAgent',
             'CDQN': 'CDQNAgent',
             'CDSarsaN': 'CDSNAgent'}


def process_observation(env_name, raw_obs):
    if env_name == 'pendulum':
        obs = raw_obs
        obs[2] /= 10
    elif env_name == 'catch':
        obs = raw_obs.flatten()
    elif env_name == 'taxi':
        obs = np.zeros(500)
        obs[raw_obs] = 1
    elif env_name == 'AC':
        obs = np.zeros(44)
        obs[raw_obs] = 1
    else:
        obs = raw_obs
    return obs


def log_data(interval, current_timestep, current_run,
             exp_type, log, env, agent, centered_values, save_weights,
             nonlinear, agent_name, bias):
    index = current_timestep // interval

    if exp_type == 'prediction':
        weights = agent.weights
        remove_offset = True
        if agent.gamma < 1:
            # weights = agent.weights + agent.avg_reward / (1 - agent.gamma) - np.dot(centered_values['d_pi'], agent.weights)
            weights += agent.avg_reward / (1 - agent.gamma)
            remove_offset = False
        rmsve = helpers.compute_rmsve(features=env.X,
                                      weights=weights,
                                      targets=centered_values['v_pi'],
                                      weighting=centered_values['d_pi'],
                                      remove_offset=remove_offset,
                                      bias=bias)
        rre = helpers.compute_rre(reward_rate_estimate=agent.avg_reward,
                                  true_reward_rate=centered_values['r_pi'])
        log['rmsve'][current_run][index] = rmsve
        log['rre'][current_run][index] = rre
    elif exp_type == 'control':
        pass

    if save_weights:
        if nonlinear:
            # agent.save_trained_model(f'{exp_name}_{exp_id}_{current_run}_ckpt_{index}')
            pass
        else:
            log['weights'][current_run][index] = agent.weights
        log['avgrew'][current_run][index] = agent.avg_reward


def save_final_weights(nonlinear, run_idx, log, env, agent, exp_name, exp_id):
    if nonlinear:
        agent.save_trained_model(f'{exp_name}_{exp_id}_{run_idx}')
    else:
        log['weights_final'][run_idx] = agent.weights
    if hasattr(agent, "avg_reward"):
        log['avgrew_final'][run_idx] = agent.avg_reward
    if hasattr(env, "best_action_count"):
        log['best_action_count'][run_idx] = env.best_action_count


def clean_up(nonlinear, location, exp_name, exp_id):
    """Zips the saved (non-linear) models of a particular param configuration."""
    if nonlinear:
        prefix = f'{location}{exp_name}_{exp_id}'
        with zipfile.ZipFile(f'{prefix}.zip', 'w') as myzip:
            for f in glob.glob(f'{location}{exp_name}_{exp_id}_*'):
                idx = f.rindex('/')
                myzip.write(filename=f, arcname=f[idx+1:])
                os.remove(f)
            print(f'Zipped the saved models to {prefix}.zip')


def print_experiment_summary(log, exp_type):
    if exp_type == 'prediction':
        tqdm.write(f'RMSVE_TVR_total\t= {np.mean(log["rmsve"]):.3f}')
        tqdm.write(f'RMSVE_TVR_lasthalf\t= {np.mean(log["rmsve"][:, log["rmsve"].shape[1] // 2:]):.3f}')
        tqdm.write(f'RRE_total\t= {np.mean(log["rre"]):.3f}')
        tqdm.write(f'RRE_lasthalf\t= {np.mean(log["rre"][:, log["rre"].shape[1] // 2:]):.3f}\n')
    elif exp_type == 'control':
        tqdm.write(f'\nRewardRate_total\t= {np.mean(log["reward"]):.3f}')
        tqdm.write(f'RewardRate_last50%\t= {np.mean(log["reward"][:, log["reward"].shape[1] // 2:]):.3f}')
        tqdm.write(f'RewardRate_last10%\t= {np.mean(log["reward"][:, log["reward"].shape[1] // 10 * 9:]):.3f}\n')


def run_experiment_one_config(config):
    """
    Runs N independent experiments for a particular parameter configuration.

    Args:
        config: a dictionary of all the experiment parameters
    Returns:
        log: a dictionary of quantities of interest
    """
    exp_name = config['exp_name']
    exp_type = config['exp_type']
    env_name = config['env_name']
    agent_name = config['agent_name']
    num_runs = config['num_runs']
    max_steps = config['num_max_steps']
    eval_every_n_steps = config['eval_every_n_steps']
    # ckpt_frequency = config.get('ckpt_frequency', eval_every_n_steps)
    save_weights = config.get('save_weights', 0)
    num_weights = config['num_weights']
    # save_counts = config.get('save_visitation_counts', False)
    csuite_env = config['csuite_env']
    render = config.get('render', False)
    nonlinear = config.get('nonlinear', False)
    reward_offset = config.get('reward_offset', 0)
    store_max_action_values = config.get('store_max_action_values', False)
    bias = config.get('bias', False)

    log = {'reward': np.zeros((num_runs, max_steps + 1), dtype=np.float32),
           # 'action': np.zeros((num_runs, max_steps + 1), dtype=np.float32),
           'weights_final': np.zeros((num_runs, num_weights), dtype=np.float32),
           'avgrew_final': np.zeros(num_runs, dtype=np.float32),
           'best_action_count': np.zeros(num_runs, dtype=np.int32)
           }
    if save_weights:
        log['avgrew'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1), dtype=np.float32)
        if not nonlinear:
            log['weights'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1,
                                       num_weights), dtype=np.float32)
            # log['trace'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1,
            #                            num_weights), dtype=np.float32)
    centered_values = []
    if exp_type == 'prediction':
        log['rmsve'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1), dtype=np.float32)
        log['rre'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1), dtype=np.float32)
        centered_values = helpers.get_centered_values(env_map[env_name], config)
    elif exp_type == 'control':
        if store_max_action_values:
            log['max_value_per_step'] = np.zeros((num_runs, max_steps // 10 + 1), dtype=np.float32)

    assert env_name in env_map, f'{env_name} not found.'
    assert agent_name in agent_map, f'{agent_name} not found.'

    for run in tqdm(range(num_runs)):
        config['rng_seed'] = run
        agent = getattr(sys.modules[__name__], agent_map[agent_name])(**config)
        if csuite_env:
            settings = {}
            if env_name == 'catch':
                # non-linear FA with 50-d binary observations and linear FA with 3-d continuous observations
                settings['observation_type'] = 'discrete' if nonlinear else 'continuous'
            env = csuite.load(env_map[env_name], settings)
            obs = env.start(seed=config['rng_seed'])
        else:
            env = getattr(sys.modules[__name__], env_map[env_name])(**config)
            obs = env.start()
        # print(f'obs: {obs}')
        action = agent.start(process_observation(env_name, obs))
        viewer = None
        if render:
            viewer = rendering.SimpleImageViewer()

        for t in range(max_steps + 1):
            # print(f'timestep: {t}')
            if render:
                viewer.imshow(env.render())
                time.sleep(0.04)
            # logging relevant data at regular intervals
            if t % eval_every_n_steps == 0:
                log_data(interval=eval_every_n_steps, current_timestep=t,
                         current_run=run, exp_type=exp_type, log=log,
                         env=env, agent=agent, centered_values=centered_values,
                         save_weights=save_weights, nonlinear=nonlinear,
                         agent_name=agent_name,
                         bias=bias)
            # the environment and agent step
            if csuite_env:
                next_obs, reward = env.step(action)
            else:
                reward, next_obs = env.step(action)
            reward += reward_offset
            # print(f'action: {action}\nreward: {reward}\nobs: {next_obs}')
            # print(f'reward: {reward:.3f}, r-bar: {agent.avg_reward:.3f}')
            action = agent.step(reward, process_observation(env_name, next_obs))
            # logging the reward at each step
            log['reward'][run][t] = reward
            # log['action'][run][t] = agent.past_action
            # log['trace'][run][t] = agent.trace
            if store_max_action_values and t > 9 * max_steps // 10:
                log['max_value_per_step'][run][t - 9*(max_steps//10)] = agent.max_value_per_step

        if render:
            viewer.close()

        save_final_weights(nonlinear=nonlinear,
                           run_idx=run, log=log, env=env, agent=agent,
                           exp_name=exp_name, exp_id=config['exp_id'])

    print_experiment_summary(log, exp_type)
    clean_up(nonlinear, config['output_folder'] + 'models/', exp_name, config['exp_id'])
    return log
