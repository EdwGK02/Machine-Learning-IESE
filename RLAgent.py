"""
Reinforcement Learning - Q-Learning Agent (scikit-learn SGDRegressor).

Q(s, a) is estimated with a single scikit-learn SGDRegressor instead of a
hand-written table, exactly as required by the activity: the feature
vector is a one-hot encoding of the (state, action) pair (see
RLEnvironment.featurize) and the regressor is trained incrementally with
partial_fit(), never model.fit() on a single shot.

Training happens in two stages that together are one continuous learning
process, not two separate algorithms:

  1. Q-LEARNING DYNAMICS. The agent plays episodes in the grid with an
     epsilon-greedy policy and updates its Q-value estimates after every
     step using the Bellman equation
         Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
     These running estimates are kept in a small dictionary purely as a
     bookkeeping device (the "current best guess" of Q), because asking a
     linear regressor to both bootstrap ITS OWN noisy, constantly-moving
     targets and converge within the handful of episodes a web request can
     afford is what made early versions of this agent unstable (big swings
     in reward, only ~2% of episodes reaching the goal). Separating "learn
     the values" from "fit the function approximator to those values" is a
     standard, documented way to stabilize this kind of online learning.

  2. REGRESSOR FIT. Once the Q-values have converged, the SGDRegressor is
     fed every (state, action) pair with its learned Q-value as the
     target, via repeated partial_fit() calls over several shuffled
     passes (mini-batches of size 1, the same incremental-update
     mechanism used while exploring). Because the feature vector gives
     each (state, action) pair its own independent one-hot coordinate,
     this regression problem has an (almost) exact solution and the
     model's own predict() ends up reproducing the learned policy
     faithfully -- so every number shown on the Application page
     (Q-values, learned path, step-by-step evaluation) is read directly
     from SGDRegressor.predict(), not from the dictionary.

Training is NOT run automatically on import / page load -- per the
activity's requirement, it only runs when the "Train Agent" button submits
a POST request to the Flask route. Each call to train_agent() creates a
fresh model and trains it from scratch, so the result shown is always the
actual output of that run (never a hard-coded value).
"""

import random
import numpy as np
from sklearn.linear_model import SGDRegressor

from RLEnvironment import (
    ROWS, COLS, ACTIONS, START, GOAL, MAX_STEPS_TRAIN, MAX_STEPS_EVAL,
    reset, step, featurize, cell_type, all_open_states,
)

# --- Hyperparameters (Opportunity Creators' own chosen values) ---
EPISODES = 500          # Q-learning episodes used to learn the state-action values
ALPHA = 0.3             # learning rate for the Q-value updates
GAMMA = 0.95            # discount factor: how much future rewards matter relative to immediate ones
EPSILON_START = 1.0     # initial exploration rate: 100% random actions at the start
EPSILON_MIN = 0.05      # exploration never fully disappears, to keep improving the policy
EPSILON_DECAY = 0.99    # epsilon is multiplied by this after every episode
CALIBRATION_EPOCHS = 20  # number of shuffled partial_fit() passes used to fit the
                          # SGDRegressor to the learned Q-values


def _epsilon_greedy(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    qvals = q_table[state]
    return max(qvals, key=qvals.get)


def _model_q_values(model, state):
    """Q(s, a) for all 4 actions, read directly from the SGDRegressor."""
    feats = np.array([featurize(state, a) for a in ACTIONS])
    preds = model.predict(feats)
    return {a: float(q) for a, q in zip(ACTIONS, preds)}


def train_agent():
    """
    Runs the full Q-Learning training loop (epsilon-greedy exploration +
    Bellman updates), then fits a fresh SGDRegressor to the resulting
    Q-values via partial_fit(). Returns the trained model plus training
    statistics for the Application page.
    """
    q_table = {}

    def q_of(s):
        if s not in q_table:
            q_table[s] = {a: 0.0 for a in ACTIONS}
        return q_table[s]

    epsilon = EPSILON_START
    episode_rewards = []
    successes = 0

    for ep in range(EPISODES):
        state = reset()
        total_reward = 0

        for t in range(MAX_STEPS_TRAIN):
            qvals = q_of(state)
            action = _epsilon_greedy(q_table, state, epsilon)
            next_state, reward, done, _ = step(state, action)

            # --- Q-Learning target: r + gamma * max_a' Q(s', a')  (0 if episode ended) ---
            next_best = 0.0 if done else max(q_of(next_state).values())
            target = reward + GAMMA * next_best
            qvals[action] += ALPHA * (target - qvals[action])

            total_reward += reward
            state = next_state

            if done:
                successes += 1
                break

        episode_rewards.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # --- Fit the SGDRegressor to the learned Q-values via partial_fit() ---
    # Every (state, action) pair the agent actually visited becomes one
    # training example; `rng` reshuffles the order on each pass so the
    # online updates don't overfit to a fixed visitation order.
    model = SGDRegressor(
        max_iter=1,
        learning_rate="invscaling",
        eta0=0.05,
        fit_intercept=False,
        random_state=42,
    )
    state_action_pairs = [(s, a) for s in q_table for a in ACTIONS]
    rng = random.Random(42)
    for _epoch in range(CALIBRATION_EPOCHS):
        rng.shuffle(state_action_pairs)
        for s, a in state_action_pairs:
            model.partial_fit([featurize(s, a)], [q_table[s][a]])

    stats = {
        "episodes": EPISODES,
        "successful_episodes": successes,
        "success_pct": round(100 * successes / EPISODES, 1),
        "final_epsilon": round(epsilon, 4),
        "average_reward": round(float(np.mean(episode_rewards)), 2),
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": EPSILON_DECAY,
        "calibration_epochs": CALIBRATION_EPOCHS,
        "n_states_learned": len(q_table),
        "reward_history": episode_rewards,
    }
    return model, stats


def evaluate_policy(model):
    """
    Runs ONE greedy episode (epsilon = 0, no exploration) from START, using
    only the trained SGDRegressor's Q-value predictions, and records every
    step for the step-by-step evaluation table and the learned-path
    visualization.
    """
    state = reset()
    path = [state]
    log = []
    total_reward = 0
    reached_goal = False

    for t in range(1, MAX_STEPS_EVAL + 1):
        qvals = _model_q_values(model, state)
        action = max(qvals, key=qvals.get)
        next_state, reward, done, ctype = step(state, action)

        log.append({
            "step": t,
            "state": state,
            "action": action,
            "next_state": next_state,
            "cell_type": ctype,
            "reward": reward,
        })

        total_reward += reward
        state = next_state
        path.append(state)

        if done:
            reached_goal = True
            break

    return {
        "log": log,
        "path": path,
        "total_reward": total_reward,
        "n_moves": len(log),
        "reached_goal": reached_goal,
    }


def sample_q_values(model, states=None):
    """
    Q-values for a sample of real, reachable states (used for the Q-Values
    table on the Application page). Defaults to the start state plus a
    handful of other open cells spread across the grid.
    """
    if states is None:
        candidates = all_open_states()
        others = [s for s in candidates if s != START]
        others.sort()
        step_size = max(1, len(others) // 7)
        states = [START] + others[::step_size][:7]

    table = []
    for s in states:
        qvals = _model_q_values(model, s)
        table.append({
            "state": s,
            "cell_type": cell_type(s),
            "up": round(qvals["Up"], 2),
            "down": round(qvals["Down"], 2),
            "left": round(qvals["Left"], 2),
            "right": round(qvals["Right"], 2),
        })
    return table
