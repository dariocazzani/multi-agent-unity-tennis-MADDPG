import random
import copy
import os
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg import DDPG
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
UPDATE_FREQUENCY = 2    # How often to update
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NUM_AGENTS = 2          # Number of agents
RANDOM_SEED = 0

class MADDPGAgent():

    def __init__(self, state_size, action_size):

        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.update_frequency = UPDATE_FREQUENCY
        self.gamma = GAMMA
        self.num_agents = NUM_AGENTS
        self.memory = ReplayBuffer()
        self.t = 0
        self.state_size = state_size
        self.action_size = action_size

        self.agents = [DDPG(index, self.state_size, self.action_size, self.num_agents, RANDOM_SEED, TAU) 
                for index in range(self.num_agents)]

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1) 
        all_next_states = all_next_states.reshape(1, -1)
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        self.t = (self.t + 1) % self.update_frequency
        if self.t == 0 and (len(self.memory) > self.batch_size):
            experiences = [self.memory.sample() for _ in range(self.num_agents)]
            self.learn(experiences, self.gamma)

    def act(self, all_states, random):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state , random=random)
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)

    def learn(self, experiences, gamma):
        all_actions = []
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(DEVICE)
            state = states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            next_state = next_states.reshape(-1, self.num_agents, self.state_size).index_select(1, agent_id).squeeze(1)
            all_actions.append(agent.actor_local(state).to(DEVICE))
            all_next_actions.append(agent.actor_target(next_state).to(DEVICE))
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)


class ReplayBuffer():

    def __init__(self):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
