import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "True"
import pygame
import numpy as np
import csuite
import copy


def convert_array_to_surface(array):
    surface = pygame.Surface(array.shape[1::-1])
    pygame.surfarray.blit_array(surface, array)
    return pygame.transform.rotate(surface, -90)

class VisualizationWindow:

    def __init__(self):
        pygame.init()
        self.shape = None
        self.screen = None

    def imshow(self, rgb_array):
        
        rgb_array = copy.copy(rgb_array)
        if self.shape == None:
            self.shape = rgb_array.shape[1::-1]
            print(self.shape)
            # print(rgb_array)
            self.screen = pygame.display.set_mode((480, 480), flags=pygame.RESIZABLE)
            print('here')
            pygame.display.set_caption("Visualization")
        
        self.screen.fill((0, 0, 0))  # Fill the screen with black
        self.screen.blit(convert_array_to_surface(rgb_array), (0, 0))
        pygame.display.update()

    def close(self):
        pygame.quit()


def test():

    env = csuite.load('pendulum')
    # env = csuite.load('half_cheetah')
    obs = env.start(seed=0)

    viewer = VisualizationWindow()

    for _ in range(100):

        action = np.random.randint(0, 3, size=(1,))
        # action = np.random.random(6)
        obs, reward = env.step(action)
        viewer.imshow(env.render())
        pygame.time.delay(60)  # Delay in milliseconds

    # pygame.init()
    # rgb_array = env.render()
    # screen = pygame.display.set_mode(rgb_array.shape[1::-1])
    # pygame.display.set_caption("Visualization")

    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     screen.fill((0, 0, 0))  # Fill the screen with black
        
    #     action = np.random.randint(0, 3, size=(1,))
    #     obs, reward = env.step(action)
    #     rgb_array = env.render()
    #     print(rgb_array.shape)
        
    #     screen.blit(convert_array_to_surface(rgb_array), (0, 0))
    #     pygame.display.update()

    #     pygame.time.delay(60)  # Delay in milliseconds

    # pygame.quit()

# from mujoco_envs import HalfCheetahContinuing
# import time
# import gymnasium as gym


# def test_rendering():
#     env = HalfCheetahContinuing(render_mode='rgb_array')
#     obs = env.start(0)

#     pygame.init()
#     screen = pygame.display.set_mode((480, 480), flags=pygame.SHOWN)
#     pygame.display.set_caption("Visualization")

#     for i in range(100):
#         action = np.random.random(6)
#         obs, reward = env.step(action)
#         print(i)

#         rgb_array = copy.copy(env.render())
#         random_array = np.random.randint(0, 255, (480, 480, 3))

#         screen.fill((0, 0, 0))  # Fill the screen with black
#         screen.blit(convert_array_to_surface(random_array), (0, 0))
#         pygame.display.update()
#         pygame.time.delay(100)
#         # time.sleep(0.1)


# def test_rendering_original_human():
#     env = gym.make('HalfCheetah-v5', render_mode='human')
#     env.reset(seed=0)

#     for i in range(100):
#         action = np.random.random(6)
#         obs, reward, _, _, _ = env.step(action)

#         env.render()
#         time.sleep(0.1)


# def test_rendering_original_rgb():
#     env = gym.make('HalfCheetah-v5', render_mode='human')
#     env.reset(seed=0)

#     pygame.init()
#     screen = pygame.display.set_mode((480, 480), pygame.DOUBLEBUF)

#     for i in range(5):
#         action = np.random.random(6)
#         obs, reward, _, _, _ = env.step(action)

#         rgb_array = env.render()
#         print(rgb_array, i)

#         # screen.fill((0, 0, 0))  # Fill the screen with black
#         # screen.blit(convert_array_to_surface(rgb_array), (0, 0))
#         # pygame.display.update()
#         # pygame.time.delay(100)

#         time.sleep(0.6)
        

# if __name__ == '__main__':
#     # test()
#     # test_rendering()
#     # test_rendering_original_human()
#     test_rendering_original_rgb()
