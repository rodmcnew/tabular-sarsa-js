# Tabular SARSA
A tabular implementation of the SARSA reinforcement learning algorithm. This agent works by building a table that can predict the expected value of every possible action in every possible environment state.

More info about SARSA: https://en.wikipedia.org/wiki/State-Action-Reward-State-Action

Usage:
```Javascript
var sarsaAgent = new tabularSarsa.Agent(numberOfPossibleStates, numberOfPossibleActions);
var lastReward = null;

function tick() {
    //Tell the agent about the current environment state and have it choose an action to take
    var action = sarsaAgent.decide(lastReward, environment.getCurrentState());

    //Tell the environment the agent's action and have it calculate a reward
    lastReward = environment.takeAction(action);
}
```

Internal details beyond basic SARSA:
- Follows an "Epsilon Greedy" exploration policy
- Uses the expected next-action reward rather than the actual next-action reward
- Uses the first seen reward as the Q value for each state-action
- Records experience replays and learns from them
- Optimized for high speed execution at the cost of code readability
