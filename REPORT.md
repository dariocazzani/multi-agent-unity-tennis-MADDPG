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
  
