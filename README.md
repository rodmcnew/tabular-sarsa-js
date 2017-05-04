# tabular-sarsa
A tabular implementation of the SARSA reinforcement learning algorithm which is related to Q-learning

@TODO show how to use the agent here
@TODO show how to how to save and load the agent here

Internal Bells and Whistles:
- Follows an "Epsilon Greedy" policy
- Uses the first seen reward as the "initial condition" for each state to speed up early learning
- Records and learns from experience replays
- Optimized for high speed execution at the cost of some code readability
