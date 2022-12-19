import gym
import pygame
from solver.policy_generator.policy_instances.envs.simple_arena import ActionSpace
import time
import requests
import numpy as np
import json

URL = "http://127.0.0.1:8000/get_next_action"


def encode_state(data: dict):
    new_dict = data.copy()
    for param, value in new_dict.items():
        if isinstance(value, np.ndarray):
            new_dict[param] = value.tolist()
        if isinstance(value, (np.int64, np.int32)):
            new_dict[param] = int(value)
        if isinstance(value, (np.float64, np.float32)):
            new_dict[param] = float(value)
    return new_dict


def get_bot_ation(state):
    data = {
        "session_id": "123",
        "description": "321",
        "state": encode_state(state)
    }
    response = requests.post(URL, json=data)
    return int(json.loads(response.content)['action'])


model = True
env = gym.make("policy_instances/SimpleArena-v0", render_mode="human")
state = env.reset()[0]
time.sleep(3)
terminated = False
while True:
    env.render()
    events = pygame.event.get()
    for event in events:
        action = None
        if not model:
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
                    state, reward, terminated, _, _ = env.step(action.value)
                if terminated:
                    state = env.reset()[0]
        else:
            if terminated:
                state = env.reset()[0]
                terminated = False
            else:
                state, reward, terminated, _, _ = env.step(get_bot_ation(state))
            pygame.time.wait(500)
