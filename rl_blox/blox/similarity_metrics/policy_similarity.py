import jax.numpy as jnp
import optax
from flax import nnx


def _pairwise_euclid(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Calculate a distance array between individual rows of the given matrices.

    Parameters
    ----------
    A : jnp.ndarray
        Matrix A
    B : jnp.ndarray
        Matrix B

    Returns
    -------
    euclidean distance :  jnp.ndarray
        Distance/Cost Matrix
    """
    # a^2
    A_sq = jnp.sum(A**2, axis=1)[:, None]
    # b^2
    B_sq = jnp.sum(B**2, axis=1)[None, :]
    # 2ab
    cross = A @ B.T
    # a^2 - 2ab + b^2 = (a-b)^2
    return A_sq + B_sq - 2 * cross


def layer_distance(
    layer1: nnx.Linear, layer2: nnx.Linear
) -> tuple[float, jnp.ndarray]:
    """Calculates the distance between 2 layers and gives
    the indices on how to align the second layer

    Parameters
    ----------
    layer1 : nnx.Linear
        Layer 1
    layer2 : nnx.Linear
        Layer 2

    Returns
    -------
    layer_distance : float
        Distance between layers
    sorting indicies : jnp.ndarray]
        indicies array for sorting
    """
    w1 = layer1.kernel
    w2 = layer2.kernel

    cost = _pairwise_euclid(w1, w2)

    mi, mj = optax.assignment.hungarian_algorithm(cost)

    sorted_i = jnp.argsort(mi)

    return cost[mi, mj].sum().item(), mj[sorted_i]


def get_network_distance(
    model1_layers: list[nnx.Linear], model2_layers: list[nnx.Linear]
) -> float:
    """Calculate the euclidiean distance between 2 networks (lists of linear layers)

    Parameters
    ----------
    model1_layers : list[nnx.Linear]
        linear layers of the 1st network
    model2_layers : list[nnx.Linear]
        linear layers of the 2nd network

    Returns
    -------
    euclidean distance : float
        euclidean distance of networks
    """
    model2_layers_cloned = [nnx.clone(layer) for layer in model2_layers]
    total_distance = 0
    for i in range(len(model1_layers)):
        # get layers
        layer1 = model1_layers[i]
        layer2 = model2_layers_cloned[i]
        # calculate distance
        dist, sort_i = layer_distance(layer1, layer2)
        total_distance += dist
        # swap perceptron so that the next layer has its inputs in the correct place
        if i + 1 < len(model1_layers):
            for swap_index in range(len(sort_i)):
                (
                    model2_layers_cloned[i].kernel,
                    model2_layers_cloned[i + 1].kernel,
                ) = _swap_perceptron(
                    model2_layers_cloned[i].kernel,
                    model2_layers_cloned[i + 1].kernel,
                    swap_index,
                    sort_i[swap_index],
                )
    return total_distance


def _swap_rows(arr: jnp.ndarray, n: int, m: int) -> jnp.ndarray:
    """Swap 2 rows of an array

    Parameters
    ----------
    arr : jnp.ndarray
        Array to swap rows in
    n : int
        1st row
    m : int
        2nd row

    Returns
    -------
    swapped array : jnp.ndarray
        Array with swapped rows
    """
    idx = jnp.arange(arr.shape[0])
    idx = idx.at[n].set(m).at[m].set(n)
    return arr[idx]


def _swap_collumns(arr: jnp.ndarray, n: int, m: int):
    """Swap 2 collumns of an array

    Parameters
    ----------
    arr : jnp.ndarray
        Array to swap collumns in
    n : int
        1st row
    m : int
        2nd row

    Returns
    -------
    swapped array : jnp.ndarray
        Array with swapped collumns
    """
    idx = jnp.arange(arr.shape[1])
    idx = idx.at[n].set(m).at[m].set(n)
    return arr[:, idx]


def _swap_perceptron(
    layer_weights: nnx.variablelib.Param,
    next_weights: nnx.variablelib.Param,
    i1: int,
    i2: int,
) -> tuple[nnx.variablelib.Param, nnx.variablelib.Param]:
    """Swap a perceptron in a network by swapping the according
    weights in it's and the next layer

    Parameters
    ----------
    layer_weights : nnx.variablelib.Param
        Perceptron's layer
    next_weights : nnx.variablelib.Param
        The following layer
    i1 : int
        Index of the perceptron
    i2 : int
        Goal Index

    Returns
    -------
    layer_weights : nnx.variablelib.Param
        Swapped weights of perceptrons layer
    next_weights : nnx.variablelib.Param
        Swapped weights of next layer
    """
    layer_weights = _swap_collumns(layer_weights, i1, i2)
    layer_weights = nnx.variablelib.Param(layer_weights)

    next_weights = _swap_rows(next_weights, i1, i2)
    next_weights = nnx.variablelib.Param(next_weights)

    return layer_weights, next_weights
