export class TabularSARSA {
    constructor(possibleStateCount, possibleActionCount) {
        var defaultOptions = {
            discountFactor: 0.9,//gamma
            randomActionProbability: 0.05,//epsilon
            learningRate: 0.1,//alpha
            replaysPerAction: 10,
            replayCountToStore: 5000,
            actionsBetweenRecordingNewReplays: 25,
            // tdErrorClamp: 1.0,
        };

        this._replayMemory = [];//@TODO trim this, don't store all
        this._actionsTillNextReplayRecording = 0;

        this._actionCount = possibleActionCount;
        this._stateCount = possibleStateCount;
        this._options = Object.assign(defaultOptions/*, options*/);

        this._lastScore = null;

        this._q = new Float64Array(this._stateCount * this._actionCount); //[];//new Array(Math.pow(2, 5 * 3));//@TODO allow state count as arg for higher performance?
        this._initializedQ = new Int8Array(this._stateCount * this._actionCount);
        this._lastActionWeights = new Float64Array(this._actionCount);

        this.oneMinusEpsilon = 1 - this._options.randomActionProbability;//cached calculations to increase performance
        this.epsilonDividedByActionCount = this._options.randomActionProbability / this._actionCount;//cached calculations to increase performance

        this.lastStep = {};
    }

    fromJson(json) {
        for (var i = 0, len = this._stateCount * this._actionCount; i < len; i++) {
            this._q[i] = json.q[i];
            this._initializedQ[i] = json.initializedQ[i];
        }
    }

    toJson() {
        var q = [];
        var initializedQ = [];
        for (var i = 0, len = this._stateCount * this._actionCount; i < len; i++) {
            q[i] = this._q[i];
            initializedQ[i] = this._initializedQ[i];
        }
        return {q: q, initializedQ: initializedQ};
    }

    /**
     * The SARSA algorithm with an epsilon greedy policy
     *
     * @param state
     * @param action
     * @param reward
     * @param nextState
     * @private
     */
    _learnFromStep(state, action, reward, nextState) {
        var currentStateActionKey = state * this._actionCount + action;
        var qOfCurrentStateAction = this._q[currentStateActionKey];

        if (qOfCurrentStateAction === 0.00
            && this._initializedQ[currentStateActionKey] !== 1
        ) {
            //Use first seen reward for a state-action as the initial value to speed up initial learning
            this._initializedQ[currentStateActionKey] = 1;//1 for true
            this._q[currentStateActionKey] = reward;
        }

        var options = this._options;
        var nextStateKeyPrepend = nextState * this._actionCount;
        var maxQofNextStateAction = this._q[nextStateKeyPrepend];
        var sumQofNextStateAction = this._q[nextStateKeyPrepend];
        for (var i = nextStateKeyPrepend + 1, max = nextStateKeyPrepend + this._actionCount; i < max; i++) {
            var thisValue = this._q[i];
            sumQofNextStateAction += thisValue;
            if (thisValue > maxQofNextStateAction) {
                maxQofNextStateAction = thisValue;
            }
        }

        var qOfNextStateAction = maxQofNextStateAction * this.oneMinusEpsilon
            + sumQofNextStateAction * this.epsilonDividedByActionCount;

        this._q[currentStateActionKey] = qOfCurrentStateAction +
            options.learningRate * (reward + options.discountFactor * qOfNextStateAction - qOfCurrentStateAction);
    }

    /**
     *
     * @param {AgentObservation} observation
     * @return {string} action code
     */
    /**
     * @TODO only bother with outputting actino weights if something like config.outputWeights is set
     *
     *
     * @param lastReward
     * @param state
     * @returns {string}
     */
    getAction(lastReward, state) {

        if (lastReward !== null) {
            this._learnFromStep(this.lastStep.state, this.lastStep.action, lastReward, state);

            var replayMemoryLength = this._replayMemory.length;
            if (replayMemoryLength > this._options.replaysPerAction) {
                for (var i = 0; i < this._options.replaysPerAction; i++) {
                    var replay = this._replayMemory[Math.floor(Math.random() * replayMemoryLength)];
                    this._learnFromStep(replay[0], replay[1], replay[2], replay[3]);
                }
            }

            this._actionsTillNextReplayRecording--;
            if (this._actionsTillNextReplayRecording < 1) {
                this._actionsTillNextReplayRecording = this._options.actionsBetweenRecordingNewReplays;
                //Trim down the replay memory any time it gets 20% larger than its allowed size
                if (this._replayMemory.length > this._options.replayCountToStore * 1.2) {
                    this._replayMemory = this._replayMemory.slice(-1 * this._options.replayCountToStore);
                }

                //@TODO only store replayes every so often
                this._replayMemory.push([this.lastStep.state, this.lastStep.action, lastReward, state]);
            }
        }


        var action;
        let actionWasRandom = false;
        if (Math.random() < this._options.randomActionProbability) {
            //@TODO this._lastActionWeights not being set in this branch
            actionWasRandom = true;
            action = getRandomIntWithZeroMin(this._actionCount);
        } else {
            var currentStateKeyPrepend = state * this._actionCount;
            var maxQofNextStateAction = this._q[currentStateKeyPrepend];
            this._lastActionWeights[0] = maxQofNextStateAction;//The loop below skips the first one so do it here
            var indexOfMaxQofNextStateAction = 0;
            var index = 0;
            for (var key = currentStateKeyPrepend + 1, max = currentStateKeyPrepend + this._actionCount; key < max; key++) {
                index++;
                var thisValue = this._q[key];
                if (thisValue > maxQofNextStateAction) {
                    maxQofNextStateAction = thisValue;
                    indexOfMaxQofNextStateAction = index;
                    this._lastActionWeights[index] = thisValue;
                }
            }
            action = indexOfMaxQofNextStateAction;
        }

        this.lastStep.state = state;
        this.lastStep.action = action;

        if (settings.renderingEnabled) {
            renderActionResponse({weights: this._lastActionWeights, wasRandom: actionWasRandom});
            renderReward(lastReward);
            // renderQTableSize(Object.keys(this._q).length);
            // renderObservationKey(state);
            // renderAdjustmentValue(adjustment.toFixed(2));
        }

        // if (typeof action === 'undefined') {
        //     throw new Error('action is undefined');
        // }

        return action;
    }
}
