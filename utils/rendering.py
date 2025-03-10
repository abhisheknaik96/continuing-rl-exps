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

    def close():
        pygame.quit()


def test():

    # env = csuite.load('pendulum')
    env = csuite.load('half_cheetah')
    obs = env.start(seed=0)

    viewer = VisualizationWindow()

    for _ in range(100):

        # action = np.random.randint(0, 3, size=(1,))
        action = np.random.random(6)
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



if __name__ == '__main__':
    test()
