var markovDecisionProcess = {
    numberOfPossibleStates: 4,
    numberOfPossibleActions: 2,
    states: [
        {// State 0
            actions: [
                {// Action 0
                    reward: 1,
                    nextState: 2
                },
                {// Action 1
                    reward: -2,
                    nextState: 3
                },
            ]
        },
        {// State 1
            actions: [
                {// Action 0
                    reward: -7,
                    nextState: 2
                },
                {// Action 1
                    reward: 100,
                    nextState: 3
                },
            ]
        },
        {// State 2
            actions: [
                {// Action 0
                    reward: 1,
                    nextState: 0
                },
                {// Action 1
                    reward: -5,
                    nextState: 3
                },
            ]
        },
        {// State 3
            actions: [
                {// Action 0
                    reward: -10,
                    nextState: 1
                },
                {// Action 1
                    reward: -10,
                    nextState: 0
                },
            ]
        }
    ]
};

class MdpEnvironment {
    constructor(mdp) {
        this._mdp = mdp;
        this._state = Math.floor(Math.random() * Object.keys(mdp.numberOfPossibleStates).length)
    }

    getCurrentState() {
        return this._state;
    }

    takeAction(action) {
        var actionInfo = this._mdp.states[this._state].actions[action];
        this._state = actionInfo.nextState;
        return actionInfo.reward;
    }
}

module.exports.numberOfPossibleStates = markovDecisionProcess.numberOfPossibleStates;
module.exports.numberOfPossibleActions = markovDecisionProcess.numberOfPossibleActions;
module.exports.Environment = function () {
    return new MdpEnvironment(markovDecisionProcess);
};
