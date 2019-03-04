import os
import math
import numpy as np
import torch

from unityagents import UnityEnvironment
import numpy as np
from collections import deque

from maddpg import MADDPGAgent
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print('The state for the second agent looks like:', states[1])

def play(agent, games=10):
    print("Playing {} games...".format(games))
    for g in range(games):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        done = False
        scores = np.zeros(num_agents) 
        while not done:
            env_info = env.step(agent.act(state, random=0))[brain_name]
            state = env_info.vector_observations
            scores += env_info.rewards
            done = any(env_info.local_done)
        print('Score (max over agents) from game {}: {:.3f}'.format(g, np.max(scores)))

def train(agent, num_episodes, max_steps):
    all_scores = []
    last_100_max_scores = deque(maxlen=100)
    avg_scores = []
    
    try:
        for episode in range(1, num_episodes):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations
            
            # Encourage exploration at the beginning
            rand = max(0.1, 1.0 - (episode / (num_episodes / 4)))

            # initialize the score (for each agent)
            scores = np.zeros(num_agents)                          
            for _ in range(max_steps):
                action = agent.act(state, rand)

                env_info = env.step(action)[brain_name]
                next_state = env_info.vector_observations
                reward = env_info.rewards
                done = env_info.local_done
                agent.step(state, action, reward, next_state, done)
                state = next_state
                scores += env_info.rewards
                if any(done):
                    break

            last_100_max_scores.append(np.max(scores))
            all_scores.append(np.max(scores))

            if len(last_100_max_scores) == last_100_max_scores.maxlen:
                avg_scores.append(np.mean(last_100_max_scores))   
                end = '\n' if (episode % 100) == 0 else '\r'     
                print("Episode: {} - Average: {:10.4f} - Best Average: {:10.4f}".format(
                        episode, avg_scores[-1], max(avg_scores)), end=end)
                
                if avg_scores[-1] >= 0.5:
                    for idx, a in enumerate(agent.agents):
                        print("\nSaving agent actor: {}".format(idx))
                        torch.save(a.actor_local.state_dict(), "./trained_models/{:.2f}_actor_{}.pt".format(avg_scores[-1], idx))
                    break
            else:
                avg_scores.append(0)

    except KeyboardInterrupt:
        print("\nManual interrupt...")
        for idx, a in enumerate(agent.agents):
            print("Saving agent actor: {}".format(idx))
            torch.save(a.actor_local.state_dict(), "./trained_models/{:.2f}_actor_{}.pt".format(avg_scores[-1], idx))

    plt.plot(range(len(all_scores)), all_scores, label="Agents' reward")
    plt.plot(range(len(all_scores)), avg_scores, label="Agents' average reward over last 100 episodes")
    plt.xlabel('Episodes', fontsize=18)
    plt.ylabel('Reward', fontsize=18)
    plt.legend(loc='best', shadow=True, fancybox=True)
    plt.show()

if __name__ == "__main__":
    agent = MADDPGAgent(state_size, action_size)
    train(agent, 10000, 200)
    play(agent)