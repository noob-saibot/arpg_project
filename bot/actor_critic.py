import torch
import torch.nn as nn
from torch.distributions import Categorical

from typing import NamedTuple
from itertools import chain

import numpy as np


class ObsSpace(NamedTuple):
    agent: np.ndarray
    agent_direction: int
    target: np.ndarray
    velocity: int


def unpack_state(state):
    state = ObsSpace(**state[0] if isinstance(state, tuple) else state)
    unpack_state = list(chain(state.agent, state.target, [state.velocity, state.agent_direction]))
    state = torch.Tensor(unpack_state).float().unsqueeze(0).to('cpu')
    return state


# class for actor-critic network
class ActorCriticNetwork(nn.Module):

    def __init__(self, obs_space, action_space):
        '''
        Args:
        - obs_space (int): observation space
        - action_space (int): action space

        '''
        super(ActorCriticNetwork, self).__init__()
        self.action_space = action_space
        # self.action_std = action_std

        self.actor = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=1))

        self.critic = nn.Sequential(
            nn.Linear(obs_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

    def forward(self):
        ''' Not implemented since we call the individual actor and critc networks for forward pass
        '''
        raise NotImplementedError

    def select_action(self, state):
        ''' Selects an action given current state
        Args:
        - network (Torch NN): network to process state
        - state (Array): Array of action space in an environment

        Return:
        - (int): action that is selected
        - (float): log probability of selecting that action given state and network
        '''

        # convert state to float tensor, add 1 dimension, allocate tensor on device
        state = unpack_state(state)

        # use network to predict action probabilities
        action_probs = self.actor(state)

        # sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        # return action
        return action.item(), m.log_prob(action)

    def evaluate_action(self, states, actions):
        ''' Get log probability and entropy of an action taken in given state
        Args:
        - states (Array): array of states to be evaluated
        - actions (Array): array of actions to be evaluated

        '''

        # convert state to float tensor, add 1 dimension, allocate tensor on device
        states_tensor = torch.stack([unpack_state(state) for state in states]).squeeze(1)

        # use network to predict action probabilities
        action_probs = self.actor(states_tensor)

        # get probability distribution
        m = Categorical(action_probs)

        # return log_prob and entropy
        return m.log_prob(torch.Tensor(actions)), m.entropy()
