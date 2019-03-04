# Report

## 1. Training Code

* ### The repository includes functional, well-documented, and organized code for training the agent. <br>
   * [x] **Script `run_training.py`**
      * From line `13` to line `35` we initialize the Unity Gym Environment
      * Function `train` at line `51` starts the training procedure. <br>
      I keep track of the scores and their avarage.
      * To encourage exploration, randomness is added to each action of agents. <br>
      At the beginning of training is almost all random uniform search, and the closer we get to 
      2500 episodes, the less randomness is used (0.1 is the minimum randomness used at training time)
      * The rest of the function is used for keeping track of the results, saving the best model and for plotting the scores
      * After achievin the required avarage score, we can watch the agent playing. Notice that at test time no randomness is added to the Agents' actions.
      
   * [x] **Script `model.py`**
     * Contains the architecture implementation for the `Actor` and `Critic` networks. More details in the following sections
   * [x] **Script `ddpg.py`**
     * Contains the implementation for a single agent using `DDPG`
   * [x] **Script `maddpg.py`**
     * Contains the implementation for training the agents using `MADDPG` - Multi-agent DDPG.

* ### The code is written in PyTorch and Python 3.
   * [x] **Check**
   
* ### The submission includes the saved model weights of the successful agent. 
   * [x] **The actors' weights are saved in `trained_models`**
   
## 2. Learning Algorithm

  * ### Multi-Agent DDPG
    * Bla
    * Bla
    
  * ### Neural Network Architecture and hyperparameters
    For the **Architectures** of both the `Actors` and `Critics` I chose to use only fully connected layers.<br>
    They are both defined in `model.py`
    
    **Actor**
    ```python
    class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    ```
    **Critic**
    ```python
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, num_agents, fc1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear((state_size + action_size)* num_agents, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        _input = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(_input))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    ```
  
    The structure is quite simple:
    * Actor: has 2 hidden layers with `400` and `300` neurons respectively.<br>
      The output has 2 neurons (corresponding to the action space size) and squeeze to be between `-1` and `+1` by the `tanh` op.
    * Critic: has 2 hidden layers with `400` and `300` neurons respectively.<br>
      The output has 1 neurons corresponding to the estimated value of the current `states+actions`.
      The `init` function expects to know the number of agents because the Critic has access to the whole information available:
      ```python
      ...
      def __init__(self, state_size, action_size, seed, num_agents, fc1_units=400, fc2_units=300):
      ...
      ```

    ``` bash
      Actor(
      (fc1): Linear(in_features=24, out_features=400, bias=True)
      (fc2): Linear(in_features=400, out_features=300, bias=True)
      (fc3): Linear(in_features=300, out_features=2, bias=True)
    )
    Critic(
      (fc1): Linear(in_features=52, out_features=400, bias=True)
      (fc2): Linear(in_features=400, out_features=300, bias=True)
      (fc3): Linear(in_features=300, out_features=1, bias=True)
    )
    ```
  
    The **Hyperparameters** are defined in `maddpg.py` from line `15` to line `24` and in `ddpg.py` from line `10` to line `13`

    ``` python
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
    LR_ACTOR = 1e-4         # learning rate of the actor 
    LR_CRITIC = 1e-3        # learning rate of the critic
    WEIGHT_DECAY = 0        # L2 weight decay
    ```

## 3. Plot of Rewards

   * **Training**: Training took 61 episodes to learn a policy that would receive an average score of **0.5** averaged on 100 episodes.
   Below the plot for the whole training (stopped as soon as the goal was reached) and a close-up on the last few hundred episodes
   
![Full-training](https://github.com/dariocazzani/multi-agent-unity-tennis-MADDPG/blob/master/images/full-training.png)
      
![Closeup-training](https://github.com/dariocazzani/multi-agent-unity-tennis-MADDPG/blob/master/images/closeup-training.png)

  * **Testing**: After the training is complete, we let the trained agents play against each other:
  ![trained-agents](https://github.com/dariocazzani/multi-agent-unity-tennis-MADDPG/blob/master/images/trained-agents.gif)

     

## 4. Ideas for Future Work

* **Sensitivity to initialization**: I found that the performance of the algorithm varied a lot across different runs with different random initializations. This is not satisfying since it feels like that it's matter of luck more than robustness of the algorithm. I'd like to continue working on it and make sure that the performance is comparable even when using different initializations

* **Network architectures**: For implementing the MADDPG I took inspiration from the architecture suggested in the Udacity Code base for the implementation of an agent that learns using `DDPG`: [Implementation](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py) <br>
  I'd like to experiment with different architectures (I believe that we can achieve good results with much smaller architectures).

* **Self-play**: Because the 2 actors are playing the same game (just from 2 sides of a mirror), it would be interesting to rewrap the problem as a **Zero-sum game** and use what we learned from **Alphazero**
