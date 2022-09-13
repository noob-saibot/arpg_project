import gym
import pygame
from solver.policy_generator.policy_instances.envs.simple_arena import ActionSpace

env = gym.make("policy_instances/SimpleArena-v0", render_mode="human")
env.reset()
while True:
    env.render()
    events = pygame.event.get()
    for event in events:
        action = None
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = ActionSpace.TurnL
            if event.key == pygame.K_RIGHT:
                action = ActionSpace.TurnR
            if event.key == pygame.K_UP:
                action = ActionSpace.MoveF
            if event.key == pygame.K_DOWN:
                action = ActionSpace.MoveB
            if event.key == pygame.K_f:
                action = ActionSpace.Touch
            if action:
                print(action)
                print(env.step(action.value))
