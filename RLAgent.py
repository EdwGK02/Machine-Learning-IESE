import random
import numpy as np
from sklearn.linear_model import SGDRegressor

from RLEnvironment import (
    ROWS, COLS, ACTIONS, START, GOAL, MAX_STEPS_TRAIN, MAX_STEPS_EVAL,
    reset, step, featurize, cell_type, all_open_states,
)


EPISODES = 500          
ALPHA = 0.3             
GAMMA = 0.95            
EPSILON_START = 1.0     
EPSILON_MIN = 0.05     
EPSILON_DECAY = 0.99    
CALIBRATION_EPOCHS = 20  
                         


def _epsilon_greedy(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    qvals = q_table[state]
    return max(qvals, key=qvals.get)


def _model_q_values(model, state):
    feats = np.array([featurize(state, a) for a in ACTIONS])
    preds = model.predict(feats)
    return {a: float(q) for a, q in zip(ACTIONS, preds)}


def train_agent():
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
