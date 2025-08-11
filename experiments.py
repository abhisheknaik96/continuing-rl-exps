"""This files runs an experiment for a particular parameter configuration."""

import time
import sys
import os
import zipfile
import glob
from tqdm import tqdm
import numpy as np
import torch
import gymnasium as gym
from utils import rendering
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
           'AO': 'AO-v0',
           'MCC': 'Continuous_MountainCarEnv',
        #    'MCC': 'MountainCarContinuous-v0',
            'pendulum_continuous': 'pendulum_continuous',
            'puckworld_continuous': 'puckworld_continuous',
            'puckworld_continuous_1d': 'puckworld_continuous_1d',
            'mujoco_swimmer': 'swimmer',
            'mujoco_half_cheetah': 'half_cheetah',
            'mujoco_ant': 'ant',
            'mujoco_humanoid': 'humanoid',
            'mujoco_reacher': 'reacher',
            'mujoco_pusher': 'pusher',
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
             'CDSarsaN': 'CDSNAgent',
             'CD_DDPG': 'DDPGAgent',
             'CD_PPO': 'PPOAgent',
             'CD_MDPO': 'MDPOAgent',
             'CD_SAC': 'SACAgent',
             'CD_TD3': 'TD3Agent',
            }


def process_observation(env_name, raw_obs):
    if env_name == 'MCC':
        obs = np.array([((raw_obs[0] + 1.2)/1.8 - 0.5) * 2, ((raw_obs[1] + 0.07)/0.14 - 0.5) * 2])
    elif env_name == 'pendulum':
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


def save_final_weights(nonlinear, run_idx, log, env, agent, exp_name, exp_id, eval_mode):
    if not eval_mode:
        if nonlinear:
            agent.save_trained_model(f'{exp_name}_{exp_id}_{run_idx}')
        else:
            log['weights_final'][run_idx] = agent.weights
        if hasattr(agent, "avg_reward"):
            log['avgrew_final'][run_idx] = agent.avg_reward
        if hasattr(env, "best_action_count"):
            log['best_action_count'][run_idx] = env.best_action_count
        if hasattr(agent, "obs_mean"):
            log['misc'][run_idx]['obs_mean'] = agent.obs_mean
        if hasattr(agent, "obs_m2"):
            log['misc'][run_idx]['obs_std'] = np.sqrt(agent.obs_m2 / agent.timestep)


def clean_up(nonlinear, location, exp_name, exp_id, eval_mode):
    """Zips the saved (non-linear) models of a particular param configuration."""
    if nonlinear and not eval_mode:
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
    store_values = config['store_values']
    # save_counts = config.get('save_visitation_counts', False)
    env_type = config['env_type']
    render = config.get('render', False)
    nonlinear = config.get('nonlinear', False)
    reward_offset = config.get('reward_offset', 0)
    store_max_action_values = config.get('store_max_action_values', False)
    bias = config.get('bias', False)
    eval_mode = config.get('eval_mode', False)
    device = config.get('device', 'cpu')
    torch.set_default_device(device)

    log = {'reward': np.zeros((num_runs, max_steps + 1), dtype=np.float32),
           'weights_final': np.zeros((num_runs, num_weights), dtype=np.float32),
           'avgrew_final': np.zeros(num_runs, dtype=np.float32),
           'misc': [{} for i in range(num_runs)]
           }
    if store_values:
        log['values'] = np.zeros((num_runs, max_steps + 1), dtype=np.float32)
        log['avgrew'] = np.zeros((num_runs, max_steps + 1), dtype=np.float32)
    if save_weights:
        if not nonlinear:
            log['weights'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1,
                                       num_weights), dtype=np.float32)
    centered_values = []
    if exp_type == 'prediction':
        log['rmsve'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1), dtype=np.float32)
        log['rre'] = np.zeros((num_runs, max_steps // eval_every_n_steps + 1), dtype=np.float32)
        centered_values = helpers.get_centered_values(env_map[env_name], config)
    elif exp_type == 'control':
        if store_max_action_values:
            log['max_value_per_step'] = np.zeros((num_runs, max_steps // 10 + 1), dtype=np.float32)

    helpers.register_continuing_mujoco_environments()
    assert env_name in env_map, f'{env_name} not found.'
    assert agent_name in agent_map, f'{agent_name} not found.'

    for run in tqdm(range(num_runs)):
        config['rng_seed'] = run
        agent = getattr(sys.modules[__name__], agent_map[agent_name])(**config)
        if env_type == 'csuite':
            settings = {}
            if env_name == 'catch':
                # non-linear FA with 50-d binary observations and linear FA with 3-d continuous observations
                settings['observation_type'] = 'discrete' if nonlinear else 'continuous'
            if render:
                settings['render_mode'] = 'human'
            env = csuite.load(env_map[env_name], settings)
            obs = env.start(seed=config['rng_seed'])
        elif env_type == 'gym':
            render_mode = 'human' if render else None
            env = gym.make(env_map[env_name], render_mode=render_mode)
            obs = env.reset()[0]
        else:
            env = getattr(sys.modules[__name__], env_map[env_name])(**config)
            obs = env.start()
        action = agent.start(process_observation(env_name, obs))
        viewer = None
        if render:
            viewer = rendering.VisualizationWindow()

        for t in range(max_steps + 1):
            if render:
                if env_type == 'csuite' and 'mujoco' not in env_name:
                    viewer.imshow(env.render())
                else:
                    env.render()        # mujoco envs will use the default gymnasium renderer
                time.sleep(0.06)
            # logging relevant data at regular intervals
            if t % eval_every_n_steps == 0:
                log_data(interval=eval_every_n_steps, current_timestep=t,
                         current_run=run, exp_type=exp_type, log=log,
                         env=env, agent=agent, centered_values=centered_values,
                         save_weights=save_weights, nonlinear=nonlinear,
                         agent_name=agent_name,
                         bias=bias)
            # the environment and agent step
            if env_type == 'csuite':
                next_obs, reward = env.step(action)
            elif env_type == 'gym':
                next_obs, reward, terminated, _, _ = env.step(action[0])
                if terminated:
                    next_obs = env.reset()[0]
            else:
                reward, next_obs = env.step(action)
            reward += reward_offset
            action = agent.step(reward, process_observation(env_name, next_obs))
            # logging the reward at each step
            log['reward'][run][t] = reward
            if store_values:
                log['values'][run][t] = agent.curr_values
                log['avgrew'][run][t] = agent.avg_reward

        if render:
            viewer.close()

        save_final_weights(nonlinear=nonlinear, eval_mode=eval_mode,
                           run_idx=run, log=log, env=env, agent=agent,
                           exp_name=exp_name, exp_id=config['exp_id'])

    print_experiment_summary(log, exp_type)
    # clean_up(nonlinear, config['output_folder'] + 'models/', exp_name, config['exp_id'], eval_mode)
    return log
