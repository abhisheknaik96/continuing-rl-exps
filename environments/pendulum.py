# Adapted from openai-gym implementation to an RL-glue interface
# by Abhishek Naik on May 3rd 2021
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
#
# ToDo: introduce damping (i.e., do not assume experiment is conducted in vacuum)

import gym
from gym import spaces
import numpy as np
from os import path


def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


class Pendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **env_args):
        self.num_states = 3      # actually, num_features
        self.num_actions = 3

        self.max_action_torque = env_args.get('max_action_torque', 2.)
        self.max_speed = env_args.get('max_speed', 8)
        self.action_noise_prob = env_args.get('action_noise_prob', 0.01)
        self.g = env_args.get('g', 10.)
        self.m = env_args.get('m', 1.)
        self.l = env_args.get('l', 1.)
        self.dt = 0.05

        self.random_seed = env_args.get('random_seed', 42)
        self.rng = np.random.RandomState(self.random_seed)
        self.render = env_args.get('render', False)
        self.count_bins = env_args.get('count_bins', (36, 16))
        self.counts = np.zeros((self.count_bins))
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_action_torque,
            high=self.max_action_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        # not much use of randomness in this domain
        self.theta = np.pi  # starts at the bottom (np.pi)
        self.theta_dot = 0  # with zero velocity (0)

        self.last_action = 0
        self.actions = np.array([-1, 0, 1]) * self.max_action_torque

    def get_noisy_action(self, chosen_action_idx):
        """with probability p returns a random action, 1-p the selection action"""
        action_idx = chosen_action_idx
        if self.rng.random() < self.action_noise_prob:
            action_idx = self.rng.choice(self.num_actions)

        return self.actions[action_idx]

    def update_counts(self):
        """updates the count for the current (theta, theta_dot) tuple"""
        theta_idx = int((self.theta + np.pi)/(2*np.pi+0.0001)*self.count_bins[0])
        theta_dot_idx = int((self.theta_dot + np.abs(self.max_speed))/(2*self.max_speed+0.0001)*self.count_bins[1])
        self.counts[theta_idx][theta_dot_idx] += 1

    def start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        observation_raw = self.get_obs()
        observation = self.process_raw_obs(observation_raw)
        # self.update_counts()

        if self.render:
            self.render_env(mode='human')
            time.sleep(0.04)

        return observation

    def step(self, action_idx: int):

        if self.render:
            self.render_env(mode='human')
            time.sleep(0.04)

        action = self.get_noisy_action(chosen_action_idx=action_idx)
        self.last_action = action

        if self.render:
            self.render_env(mode='human')

        costs = angle_normalize(self.theta) ** 2 + .1 * self.theta_dot ** 2 \
                + .001 * (action ** 2)

        # original gym dynamics
        newthdot = self.theta_dot + (-3 * self.g / (2 * self.l) * np.sin(self.theta + np.pi)
                                     + 3. / (self.m * self.l ** 2) * action) * self.dt
        newth = angle_normalize(self.theta + newthdot * self.dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        # csuite dynamics
        # newth = self.theta
        # newthdot = self.theta_dot
        # for _ in range(4):
        #     newthdot += (action + self.g * np.sin(self.theta)) * self.dt
        #     newth += newthdot * self.dt
        #
        #     newth = angle_normalize(newth)
        #     newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.theta = np.round(newth, 2)
        self.theta_dot = np.round(newthdot, 2)
        # self.update_counts()

        reward = np.round(-costs, 2)
        observation_raw = self.get_obs()
        observation = self.process_raw_obs(observation_raw)

        return reward, observation

    def get_obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])

    def process_raw_obs(self, obs_raw):
        """
        converts the dictionary into a usable list with scaled features
        """
        return obs_raw

    def render_env(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.theta + np.pi / 2)
        # if self.last_action:
        self.imgtrans.scale = (-self.last_action / 2, np.abs(self.last_action) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def env_cleanup(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


import time
# from gym.envs.classic_control import rendering


if __name__ == "__main__":

    # viewer = rendering.SimpleImageViewer()
    env = Pendulum(render=True, action_noise_prob=0.0)
    obs = env.start()
    last_obs = obs
    print(obs)
    actions = [-1, 0, 1]
    # last_action_idx = 1
    for i in range(100):
        # action_idx = np.random.choice(len(actions))
        # action_idx = 2 - last_action_idx
        # viewer.imshow(env.render())
        # time.sleep(0.04)
        action_idx = 0 if last_obs[2] < 0 else 2
        print(actions[action_idx])
        reward, obs = env.step(action_idx)
        print(reward, obs)
        last_obs = obs
