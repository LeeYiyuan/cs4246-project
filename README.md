# cs4246-project
My solution to the CS4246 project, AY 19/20 semester 1. The resulting agent came in first overall on the leaderboard, achieving a success rate of 99.86%. The runner-up came in close at 99.70%.

### Overview
This solution takes a multi-staged approach. Firstly, an _expert agent_ was handcrafted that moves to the lane above if it estimates that the probability that its resultant position will be occupied by some other car is below 0.05. Otherwise, it moves forward with speed -1, with the exception of it the agent being in the top lane, where it moves forward with the greatest possible speed leading to no collision. For each row, the estimation of occupancy for the next time step was done using an RNN that takes as input the history of the position of cars (note: excludes agent) and the occupancy trails. This expert agent achieved a success rate of ~ 83%, with the RNN having a prediction accuracy of ~ 91% per cell (or an average cross-entropy error of 0.089).

Next, the expert agent was used as a guide for deep Q-learning. A variant of epsilon-greedy was employed, where everytime a random action is desired, we select the expert agent instead. Indeed, such a guide is required since the difficulty of gaining a reward signal through completely random actions as per vanilla DQN makes it hard for the training to have a gradient for the training to descend on. epsilon is decayed overtime from 1.00 to 0.00 in a fashion similar to Homework 3 (exponential decay). The resultant DQN achieved a success rate of 76%. While this is lower than the expert agent, the significance is that we have successfully transfered the learning onto a deep neural network that can be used for deep approaches.

Then, fixed epsilon-greedy (epsilon = 0.10) was used to allow for some exploration, since the transferred model may not be optimal (it learns from the unnecessarily optimal expert agent). This improved the success rate to 97%. Finally, fully greedy deep Q-learning was used to fine-tune the weights. Here, we separate the transitions from successful episodes from those from failed episodes into two buffers. A minibatch sample is taken from each buffer, and the loss is added together with equal weight before it is backpropagated across the model. This is to ensure that the failure transitions do not get washed out by the success transitions due to the otherwise low probability of being sampled (most of the replay buffer contains success transitions), or that the model does not overfit on success transitions. This gave a final success rate of 99.86%, topping the leaderboard.

### Description of files

- `expert_agent.py`: Expert agent that uses a handcrafted policy, additionally relying on trained RNNs for 1-step and 2-step lookahead prediction of cell occupancies.
- `dqn_agent.py`: Agent that plays using a trained DQN.
- `train_rtrailnet.py`: Trains the RNN to predict cell occupancy (allows for n-step lookahead).
- `train_dqn.py`: Trains DQN using expert agent as a guide for each epsilon.
- `train_dqn_epsilon.py`: Trains DQN using epsilon-greedy.
- `train_dqn_active.py`: Trains DQN using full greedy, with equal weighting for success and failure transitions.
