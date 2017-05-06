# Tabular Expected SARSA Agent
This contains an agent that learns to maximaize reward through reinforcement learning. The agent works by building a table that can predict the expected value of every possible action from every possible state. Exploration is accomplish through an epsilon greedy policy.

Because this uses table-based Q function, it only works in environments with a discrete set of states and actions. You must be able to convert all states and actions to integers to use this agent.

Installation:
```
npm install tabular-sarsa
```

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

Optimizations beyond plain SARSA:
- Uses "Expected SARSA" rather than plain SARSA to speed up learning
- Uses the first seen reward as the initial Q value for each state-action to speed up learning
 
More info about the Expected-SARSA algorithm:
http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf
