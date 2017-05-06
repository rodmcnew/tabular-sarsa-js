class Agent {
    /**
     * @param {int} numberOfPossibleStates
     * @param {int} numberOfPossibleActions
     * @param {Object} [options]
     */
    constructor(numberOfPossibleStates, numberOfPossibleActions, options = {}) {
        this._actionCount = numberOfPossibleActions;
        this._stateCount = numberOfPossibleStates;

        this._options = Object.assign(
            {   //Default options
                learningEnabled: true, //set to false to disable all learning for higher execution speeds
                learningRate: 0.1,//alpha - how much new experiences overwrite previous ones
                explorationProbability: 0.05,//epsilon - the probability of taking random actions in the Epsilon Greedy policy
                discountFactor: 0.9,//discountFactor - future rewards are multiplied by this
            },
            options
        );

        //Stores the expected reward for a given state and action. Is a 2D table stored as a flat array for higher speed
        this._q = new Float64Array(this._stateCount * this._actionCount);

        //Stores 0 if we haven't seen a reward for this state-action before, stores 1 if we have
        this._initializedQ = new Int8Array(this._stateCount * this._actionCount);

        //Some values used in the SARSA algorithm. We pre-calculate them here for higher speed
        this._oneMinusEpsilon = 1 - this._options.explorationProbability;
        this._epsilonDividedByActionCount = this._options.explorationProbability / this._actionCount;

        //Properties used to store statistics about the last action for reporting reasons
        this._qOfLastState = new Float64Array(this._actionCount);
        this._lastActionWasRandom = false;

        //The last state and action we saw
        this._lastState = 0;
        this._lastAction = 0;
    }

    /**
     * Learn from the last reward, decide on the next action to take, and return the next action
     *
     * @param {float|null} lastReward if we are on the very first step, pass null here, otherwise pass a float
     * @param {int} state
     * @returns {int} the action that the agent decided to take
     */
    decide(lastReward, state) {
        if (lastReward !== null && this._options.learningEnabled === true) {
            //Learn from the current step
            this._learnFromStateActionRewardState(this._lastState, this._lastAction, lastReward, state);
        }

        this._lastActionWasRandom = false;

        var greatistExpectedReward = this._q[state * this._actionCount];
        var actionWithGreatistExpectedReward = 0;
        for (var actionI = 0; actionI < this._actionCount; actionI++) {
            var expectedRewardOfThisAction = this._q[state * this._actionCount + actionI]

            //Log the last action weights. Charting these can be useful
            this._qOfLastState[actionI] = expectedRewardOfThisAction;

            if (expectedRewardOfThisAction > greatistExpectedReward) {
                greatistExpectedReward = expectedRewardOfThisAction;
                actionWithGreatistExpectedReward = actionI;
            }
        }

        this._lastAction = actionWithGreatistExpectedReward;

        //Epsilon greedy exploration policy - take random exploration actios with a probability of epsilon
        if (Math.random() < this._options.explorationProbability) {
            this._lastAction = Math.floor(Math.random() * this._actionCount);
            this._lastActionWasRandom = true;
        }

        this._lastState = state;

        return this._lastAction;
    }

    /**
     * The SARSA algorithm with an epsilon greedy policy
     *
     * @param {int} state
     * @param {int} action
     * @param {float} reward
     * @param {int} nextState
     * @private
     */
    _learnFromStateActionRewardState(state, action, reward, nextState) {
        var currentStateActionKey = state * this._actionCount + action;
        var qOfCurrentStateAction = this._q[currentStateActionKey];

        if (qOfCurrentStateAction === 0.00
            && this._initializedQ[currentStateActionKey] !== 1
        ) {
            //Use first seen reward for a state-action as the initial value to speed up initial learning
            this._initializedQ[currentStateActionKey] = 1;//1 for true
            this._q[currentStateActionKey] = reward;
            return;
        }

        var nextStateKeyPrepend = nextState * this._actionCount;
        var maxQofNextStateAction = this._q[nextStateKeyPrepend];
        var sumQofNextStateActions = this._q[nextStateKeyPrepend];
        for (var i = nextStateKeyPrepend + 1, max = nextStateKeyPrepend + this._actionCount; i < max; i++) {
            var thisValue = this._q[i];
            sumQofNextStateActions += thisValue;
            if (thisValue > maxQofNextStateAction) {
                maxQofNextStateAction = thisValue;
            }
        }

        //Update the Q table by using the SARSA algorithm with an "epsilon greedy" policy
        this._q[currentStateActionKey] += this._options.learningRate * (
                reward
                + this._options.discountFactor * (
                    maxQofNextStateAction * this._oneMinusEpsilon +
                    sumQofNextStateActions * this._epsilonDividedByActionCount
                )
                - qOfCurrentStateAction
            );
    }

    /**
     * Returns some additional info about the last action that was taking. Useful for graphs and reports
     *
     * @returns {{action: (number|*), weights: Float64Array, wasRandomlyChosen: boolean}}
     */
    getLastActionStats() {
        return {
            action: this._lastAction,
            wasRandomlyChosen: this._lastActionWasRandom,
            weights: this._qOfLastState
        }
    }

    /**
     * Saves everything the agent has learned to a JSON-serializable object and returns it
     *
     * @returns {{q: Array, initializedQ: Array}}
     */
    saveToJson() {
        var q = [];
        var initializedQ = [];
        for (var i = 0, len = this._stateCount * this._actionCount; i < len; i++) {
            q[i] = this._q[i];
            initializedQ[i] = this._initializedQ[i];
        }
        return {q: q, initializedQ: initializedQ};
    }

    /**
     * Loads a previously saved agent
     *
     * @param {{q: Array, initializedQ: Array}} json
     */
    loadFromJson(json) {
        for (var i = 0, len = this._stateCount * this._actionCount; i < len; i++) {
            this._q[i] = json.q[i];
            this._initializedQ[i] = json.initializedQ[i];
        }
    }
}

module.exports.Agent = Agent;
