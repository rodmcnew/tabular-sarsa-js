//To run this demo: cd here and then run "node index.js"
var tabularSarsa = require('../src/index');
var exampleWorld = require('./world/example-world');
var sarsaAgent = new tabularSarsa.Agent(
    exampleWorld.numberOfPossibleStates,
    exampleWorld.numberOfPossibleActions
);

/**
 * Run a game where we see how high of a total reward we can get in 100 actions
 *
 * @returns {number} Total reward for this game
 */
function runGame() {
    var totalReward = 0;
    var environment = new exampleWorld.Environment();
    var lastReward = null;

    function tick() {
        //Tell the agent about the current environment state and have it choose an action to take
        var action = sarsaAgent.decide(lastReward, environment.getCurrentState());

        //Tell the environment the agent's action and have it calculate a reward
        lastReward = environment.takeAction(action);

        //Log the total reward
        totalReward += lastReward;
    }

    for (var i = 0; i < 100; i++) {
        tick();
    }

    return totalReward;
}

var gamesToRun = 10000;
var totalRewards = [];
console.log('Playing ' + gamesToRun + ' games...');
console.time('Run time');
while (gamesToRun--) {
    totalRewards.push(runGame());
}
console.timeEnd('Run time');
var first100 = totalRewards.slice(0, 10);
var last100 = totalRewards.slice(-10);
console.log(
    'Average total reward for first 10 games:',
    first100.reduce((acc, totalReward) => acc + totalReward) / first100.length
);
console.log(
    'Average total reward for last 10 games:',
    last100.reduce((acc, totalReward) => acc + totalReward) / last100.length
);
