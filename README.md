# tabular-sarsa
A tabular implementation of the SARSA reinforcement learning algorithm which is related to Q-learning

Bells and Whistles:
- Follows an "Epsilon Greedy" policy
- Uses the first seen reward as the "initial condition" for each state to speed up early learning
- Stores all Q values in a single Float64Array for high speed execution
