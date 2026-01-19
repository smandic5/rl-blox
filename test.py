import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

desc = generate_random_map(size=8, seed=1)
envs = gym.make_vec("FrozenLake-v1", 20, desc=desc)

desc = [list(s) for s in desc]


HOLE, GOAL, FROZEN, OOB = range(4)
CODE_MAP = {
    "S": FROZEN,
    "F": FROZEN,
    "H": HOLE,
    "G": GOAL,
    "O": OOB,
}


def get_code(desc, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    grid = np.asarray(desc)

    oob = (x < 0) | (y < 0) | (x >= 8) | (y >= 8)

    tiles = np.full(x.shape, "O", dtype="<U1")
    print(tiles)

    tiles[~oob] = grid[y[~oob], x[~oob]]
    print(tiles)

    indices = np.vectorize(CODE_MAP.get)(tiles)
    return np.eye(4)[indices]


OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_surrounding(desc, state):
    x = state % 8
    y = state // 8

    return np.concatenate(
        [get_code(desc, x + dx, y + dy) for dx, dy in OFFSETS],
        axis=-1,
    )


print(get_surrounding(desc, np.array([0, 8, 7])))
