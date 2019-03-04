# Report

## 1. Training Code

* ### The repository includes functional, well-documented, and organized code for training the agent. <br>
   * [x] **Script `run_training.py`**
      * From line `13` to line `35` we initialize the Unity Gym Environment
      * Function `train` at line `51` starts the training procedure. <br>
      We keep track of the scores and their avarage.
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
