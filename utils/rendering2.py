import pygame
import numpy as np
import csuite


def convert_array_to_surface(array):
    surface = pygame.Surface(array.shape[1::-1])
    pygame.surfarray.blit_array(surface, array)
    return pygame.transform.rotate(surface, -90)

env = csuite.load('pendulum')
obs = env.start(seed=0)

# for i in range(10):
#     action = env.action_space.sample()
#     obs, reward = env.step(action)
#     screen.blit(convert_array_to_surface(obs), (0, 0))
#     pygame.display.update()
#     pygame.time.delay(1000)

pygame.init()
rgb_array = env.render()
screen = pygame.display.set_mode(rgb_array.shape[1::-1])
pygame.display.set_caption("Visualization")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  # Fill the screen with black
    
    action = np.random.randint(0, 3, size=(1,))
    obs, reward = env.step(action)
    rgb_array = env.render()
    
    screen.blit(convert_array_to_surface(rgb_array), (0, 0))
    pygame.display.update()

    pygame.time.delay(60)  # Delay in milliseconds

pygame.quit()