import random, copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import Actor, Critic

RANDOM_SEED = 0
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG():

    def __init__(self, index, state_size, action_size, num_agents, random_seed, tau):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self.index = index
        self.action_size = action_size
        self.tau = tau

        self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_local = Critic(state_size, action_size, random_seed, num_agents).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed, num_agents).to(DEVICE)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    def act(self, state, random):
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(torch.from_numpy(state).float().to(DEVICE)).cpu().data.numpy()
        self.actor_local.train()
        # Interleave actions with random actions
        action = (1 - random) * action + random * np.random.uniform(-1, 1, self.action_size)
        return np.clip(action, -1, 1)

    def learn(self, index, experiences, gamma, all_next_actions, all_actions):
        states, actions, rewards, next_states, dones = experiences

        self.critic_optimizer.zero_grad()

        index = torch.tensor([index]).to(DEVICE)
        actions_next = torch.cat(all_next_actions, dim=1).to(DEVICE)
        with torch.no_grad():
            q_next = self.critic_target(next_states, actions_next)
        q_exp = self.critic_local(states, actions)
        q_t = rewards.index_select(1, index) + (gamma * q_next * (1 - dones.index_select(1, index)))
        F.mse_loss(q_exp, q_t.detach()).backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        actions_pred = [actions if i == self.index else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(DEVICE)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()

        self.actor_optimizer.step()

        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
            """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
            Params
            ======
                local_model: PyTorch model (weights will be copied from)
                target_model: PyTorch model (weights will be copied to)
                tau (float): interpolation parameter 
            """
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)