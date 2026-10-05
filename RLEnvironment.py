"""
Reinforcement Learning - Environment.

Defines the 10x10 grid-world environment used by the Q-Learning agent,
extending the 4x4 maze exercise developed in class. The environment keeps
the same four character conventions (A, T, o, #) and adds a fifth one (D,
Danger Zone) as required by the activity.

Grid legend:
    A -> Agent start position   (1 cell)
    T -> Target / goal           (1 cell)
    o -> Open / walkable path    (68 cells)
    # -> Wall / obstacle         (20 cells)
    D -> Danger zone (walkable, but penalized) (10 cells)

The grid below was generated programmatically (fixed random seed) and then
verified with a breadth-first search to guarantee: (1) every required cell
count matches the activity's specification exactly, and (2) a valid path of
at least 14 steps exists between the start and the goal, so the agent has a
non-trivial routing problem to learn.
"""

import numpy as np

ROWS, COLS = 10, 10

GRID = [
    "Ao##oooo##",
    "oo#o#oo#DD",
    "oooo#ooooo",
    "oooDoooooo",
    "oo#ooooo##",
    "ooo#D#oooo",
    "oooo#oo#oo",
    "DoDoooo#oo",
    "D#oDooo#oo",
    "o#oo#ooDDT",
]

START = (0, 0)
GOAL = (9, 9)

ACTIONS = ["Up", "Down", "Left", "Right"]
ACTION_DELTAS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

# --- Reward system (Opportunity Creators' own values, documented in the report) ---
REWARD_MOVE = -1          # moving to a valid open ("o") cell -- small cost per step,
                           # so the agent is encouraged to find the SHORTEST path.
REWARD_INVALID = -5       # attempting to move off the grid or into a wall ("#"):
                           # the agent stays in place and is penalized more than a
                           # normal move, to discourage wasting steps on illegal moves.
REWARD_DANGER = -20       # entering a Danger Zone ("D"): allowed, but costly --
                           # steep enough that the agent learns to route around
                           # danger zones whenever a safer path of similar length exists.
REWARD_GOAL = 100         # reaching the Target ("T"): large positive reward so the
                           # agent is clearly guided toward finishing the episode.

MAX_STEPS_TRAIN = 150     # episode cutoff during training
MAX_STEPS_EVAL = 150      # episode cutoff during the greedy (no-exploration) evaluation


def cell_type(pos):
    r, c = pos
    return GRID[r][c]


def in_bounds(pos):
    r, c = pos
    return 0 <= r < ROWS and 0 <= c < COLS


def is_wall(pos):
    return in_bounds(pos) and cell_type(pos) == "#"


def reset():
    """Returns the starting state."""
    return START


def step(state, action):
    """
    Applies `action` from `state` and returns (next_state, reward, done, cell_type_label).

    Rules:
      - Moving off the grid or into a wall ("#") is an invalid move: the agent
        stays in its current cell and receives REWARD_INVALID.
      - Moving onto a Danger Zone ("D") is allowed but receives REWARD_DANGER.
      - Moving onto an open cell ("o") or the start cell receives REWARD_MOVE.
      - Reaching the Target ("T") receives REWARD_GOAL and ends the episode.
    """
    dr, dc = ACTION_DELTAS[action]
    r, c = state
    next_pos = (r + dr, c + dc)

    if not in_bounds(next_pos) or is_wall(next_pos):
        return state, REWARD_INVALID, False, "Invalid move"

    ctype = cell_type(next_pos)

    if ctype == "T":
        return next_pos, REWARD_GOAL, True, "Goal"
    elif ctype == "D":
        return next_pos, REWARD_DANGER, False, "Danger Zone"
    else:  # "o" or "A" (start, walkable)
        return next_pos, REWARD_MOVE, False, "Path"


N_STATE_ACTIONS = ROWS * COLS * len(ACTIONS)


def featurize(state, action):
    """
    One-hot encodes the (state, action) PAIR into a single feature vector,
    so a scikit-learn SGDRegressor can approximate Q(s, a) with one
    independent weight per pair, trained incrementally with partial_fit().

    IMPORTANT: this must be a *joint* one-hot of (state, action), not a
    concatenation of separate one-hot blocks for row, column and action.
    A concatenated encoding forces an ADDITIVE model
    Q(s, a) = f(row) + g(col) + h(action), which cannot represent the
    irregular, cell-specific values created by scattered walls and danger
    zones (e.g. two cells in the same row/column can need very different
    Q-values once a wall or danger zone sits between them). Giving every
    (state, action) pair its own one-hot coordinate removes that
    restriction entirely: the regressor becomes mathematically equivalent
    to a lookup table, exactly like tabular Q-learning, while still being
    estimated via SGDRegressor.partial_fit() as the activity requires.

    Vector length = ROWS * COLS * len(ACTIONS) = 10 * 10 * 4 = 400.
    """
    r, c = state
    vec = np.zeros(N_STATE_ACTIONS)
    cell_index = r * COLS + c
    vec[cell_index * len(ACTIONS) + ACTIONS.index(action)] = 1.0
    return vec


def all_open_states():
    """All non-wall states (used to report Q-values for a sample of real, reachable states)."""
    return [(r, c) for r in range(ROWS) for c in range(COLS) if GRID[r][c] != "#"]
