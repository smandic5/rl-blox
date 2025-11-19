import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.optimize import linprog


def kantorovich_distance(
    d: np.ndarray, p: np.ndarray = None, q: np.ndarray = None
):
    """
    Calculates the Kantorovich distance between two probability distributions.
    The Kantorovich distance is the solution to the linear programming problem:
    min_T sum_{i,j} d_{i,j} * T_{i,j}
    s.t. sum_j T_{i,j} = p_i for all i
            sum_i T_{i,j} = q_j for all j
            d_{i,j} >= 0 for all i,j

    Params
    ------
    d_matrix: np.ndarray
        Cost matrix with shape (n, m).
    p: np.ndarray | None
        Probability distribution over the rows with shape (n,). If None, uniform distribution is used.
    q: np.ndarray | None
        Probability distribution over the columns with shape (m,). If None, uniform distribution is used.
    """
    n, m = d.shape
    if p is None:
        p = np.ones(n) / n
    if q is None:
        q = np.ones(m) / m
    c = d.flatten()
    A_eq, b_eq = [], []
    for i in range(n):  # Row sums
        row = np.zeros((n, m))
        row[i, :] = 1
        A_eq.append(row.flatten())
        b_eq.append(p[i])
    for j in range(m):  # Column sums
        col = np.zeros((n, m))
        col[:, j] = 1
        A_eq.append(col.flatten())
        b_eq.append(q[j])
    A_eq, b_eq = np.array(A_eq), np.array(b_eq)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.fun


def kantorovich_distance_ott(
    d: jnp.ndarray,
    p: jnp.ndarray = None,
    q: jnp.ndarray = None,
    epsilon: float = 1e-2,
):
    """
    Calculates Kantorovich distance using OTT-JAX.

    Parameters
    ----------
    d : jnp.ndarray
        Cost matrix with shape (n, m)
    p : jnp.ndarray | None
        Source distribution (n,). If None, uniform.
    q : jnp.ndarray | None
        Target distribution (m,). If None, uniform.
    epsilon : float
        Entropic regularization parameter. Lower = closer to exact LP.
    """
    n, m = d.shape
    if p is None:
        p = jnp.ones(n) / n
    if q is None:
        q = jnp.ones(m) / m

    geom = geometry.Geometry(cost_matrix=d, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a=p, b=q)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    print(out.reg_ot_cost)
    return out.reg_ot_cost


def bisimulation_distance(R_i, R_j, P_i, P_j, c=0.5, tol=1e-6, max_iter=1000):
    """
    Calculates the bisimulation distance between two MDPs using value iteration.

    Params
    ------
    R_i: np.ndarray
        Reward matrix of MDP i with shape (n_states, n_actions).
    R_j: np.ndarray
        Reward matrix of MDP j with shape (n_states, n_actions).
    P_i: np.ndarray
        Transition probability tensor of MDP i with shape (n_states, n_actions, n_states
    P_j: np.ndarray
        Transition probability tensor of MDP j with shape (n_states, n_actions, n_states
    c: float
        Weighting factor between reward and transition differences.
    tol: float
        Tolerance for convergence.
    max_iter: int
        Maximum number of iterations.
    """
    n_i_states, n_actions, _ = R_i.shape
    n_j_states = R_j.shape[0]
    expected_R_i, expected_R_j = expected_rewards(P_i, R_i), expected_rewards(
        P_j, R_j
    )
    d = np.zeros((n_i_states, n_j_states))
    for it in range(max_iter):
        print("Iteration:", it)
        d_new = np.zeros_like(d)
        for i in range(n_i_states):
            for j in range(n_j_states):
                vals = []
                for a in range(n_actions):
                    reward_diff = abs(expected_R_i[i, a] - expected_R_j[j, a])
                    trans_diff = kantorovich_distance_ott(
                        d, P_i[i, a], P_j[j, a]
                    )
                    vals.append((1 - c) * reward_diff + c * trans_diff)
                d_new[i, j] = max(vals)
        if np.max(np.abs(d_new - d)) < tol:
            return d_new
        d = d_new
    return d


def expected_rewards(P, R):
    return np.einsum("san,san->sa", P, R)


def get_env_model(env: gym.Env):

    env = env.unwrapped

    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))

    for s in range(n_states):
        for a in range(n_actions):
            for prob, s_next, reward, done in env.P[s][a]:
                P[s, a, s_next] += prob
                R[s, a, s_next] += reward

    return P, R


def compute_task_similarity(env1, env2, c=0.5):
    P1, R1 = get_env_model(env1)
    P2, R2 = get_env_model(env2)

    d_matrix = bisimulation_distance(R1, R2, P1, P2, c=c)
    return d_matrix


import jax

from ...multitask import WeightedTaskSelector


def get_bisimulation_distance_matrix(envs: list) -> jnp.ndarray:
    num_envs = len(envs)
    dist = jnp.zeros((num_envs, num_envs))
    for this in range(num_envs):
        for other in range(num_envs):
            if this == other:
                continue
            similarity = compute_task_similarity(envs[this], envs[other])
            dist = dist.at[this, other].set(similarity)
            dist = dist.at[other, this].set(similarity)
    return dist


class ModelBasedTaskSelector(WeightedTaskSelector):
    def __init__(
        self,
        tasks,
        envs: list = [],
        prefer_similar: bool = True,
        choose_from_last_pick: bool = False,
        **kwargs,
    ):
        super().__init__(tasks, weights=None, **kwargs)
        self.prefer_similar = prefer_similar
        self.choose_from_last_pick = choose_from_last_pick
        self.last_picked = 0

        self.cost_matrix = get_bisimulation_distance_matrix(envs)

    def recalculate_weights(self):
        if self.choose_from_last_pick:
            policy_costs = self.cost_matrix[self.last_picked]
        else:
            policy_costs = jnp.sum(self.cost_matrix, axis=-1)
        if not self.prefer_similar:
            policy_costs *= -1
        self.weights = jax.nn.softmax(policy_costs)

    def select(self):
        self.last_picked = super().select()
        return self.last_picked

    def feedback(self, reward: float, **kwargs):
        super().feedback(reward)
        self.recalculate_weights()
