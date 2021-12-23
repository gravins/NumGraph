from typing import Callable, Optional
from numpy.typing import NDArray
from numpy.random import default_rng, Generator
import numpy as np


def to_dense(edge_list: NDArray, num_nodes: int = None) -> NDArray:
    """
    Converts a list of edges in a squared adjacency matrix

    Parameters
    ----------
    edge_list : NDArray
        The list of edges (num_edges x 2)
    num_nodes : int, optional
        The number of nodes in the graph, by default None

    Returns
    -------
    NDArray
        The squared adjacency matrix (num_nodes x num_nodes)
    """
    if not num_nodes:
        num_nodes = np.max(edge_list) + 1

    dense_adj = np.zeros((num_nodes, num_nodes))

    for i, j in edge_list:
        dense_adj[i, j] = 1

    return dense_adj


def to_sparse(adj_matrix: NDArray) -> NDArray:

    """
    Converts an adjacency matrix to a list of edges

    Parameters
    ----------
    adj_matrix : NDArray
        The squared adjacency matrix (num_nodes x num_nodes)

    Returns
    -------
    NDArray
        The list of edges (num_edges x 2)
    """

    return np.argwhere(adj_matrix > 0)


def to_undirected(edge_list: NDArray) -> NDArray:
    """
    Turns a directed edge_list into a non-directed one

    Parameters
    ----------
    edge_list : NDArray
        A directed edge list (num_edges x 2)

    Returns
    -------
    NDArray
        An undirected edge list ((2*num_edges) x 2)
    """
    sources, targets = edge_list[:, 0], edge_list[:, 1]
    sources, targets = sources.reshape((-1, 1)), targets.reshape((-1, 1))

    new_edges = np.concatenate((targets, sources), axis=1)
    edge_list = np.concatenate((edge_list, new_edges), axis=0)

    return edge_list

def coalesce(edge_list: NDArray) -> NDArray:
    """
    Polishes an edge list by removing duplicates and by sorting the edges

    Parameters
    ----------
    edge_list : NDArray
        An edge list (num_edges x 2)

    Returns
    -------
    NDArray
        A sorted edge list with no duplicated edges (new_num_edges x 2)
    """
    return np.unique(edge_list, axis=0)


def dense(generator):
    """
    Transforms a sparse generator into its dense version

    Parameters
    ----------
    generator : Callable
        A callable that generates graphs

    Returns
    -------
    Callable
        A callable that generates the squared adjacency matrix (num_nodes x num_nodes) of a graph
    """
    return lambda *args, **kwargs: to_dense(generator(*args, **kwargs))


def weighted(generator: Callable, directed: bool = False, low: float = 0.0, high: float = 1.0,
             rng: Optional[Generator] = None) -> Callable:
    """
    Takes as input a graph generator and returns a new generator function that outputs weighted
    graphs. If the generator is dense, the output will be the weighted adjacency matrix. If the
    generator is sparse, the new function will return a tuple (adj_list, weights).

    Parameters
    ----------
    generator : Callable
        A callable that generates graphs
    directed: bool
        Whether to generate weights for directed graphs
    low : float, optional
        Lower boundary of the sampling distribution interval,
        i.e., interval in [low, high), by default 0.0
    high : float, optional
        Upper boundary of the sampling distribution interval,
        i.e., interval in [low, high), by default 1.0
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    Callable
        A callable that generates weighted graphs

    Examples
    --------
    >> weighted(erdos_renyi)(num_nodes=100, prob=0.5)
    """

    if rng is None:
        rng = default_rng()

    def weighted_generator(*args, **kwargs):
        adj = generator(*args, **kwargs)

        if adj.shape[0] == adj.shape[1]:
            num_nodes = adj.shape[0]
            weights = rng.uniform(low=low, high=high, size=(num_nodes, num_nodes))

            if not directed:
                weights = np.triu(weights)
                weights = weights + weights.T

            adj = adj.astype(float) * weights
            return adj

        weights = rng.uniform(low=low, high=high, size=(adj.shape[0], 1))

        return adj, weights

    return weighted_generator


def remove_self_loops(adj: NDArray) -> NDArray:
    """
    Removes every self-loop in the graph given by adj

    Parameters
    ----------
    adj : NDArray
        The adjancency matrix (num_edges x 2)

    Returns
    -------
    NDAarray
        The list of edges without self-loops (new_num_edges x 2)

    Raises
    ------
    NotImplementedError
    """
    row, col = adj.shape

    if row == col:
        # The case of a squared dense adj matrix
        return adj * (1 - np.eye(row, col, dtype=np.bool8))

    sources, targets = adj[:, 0], adj[:, 1]
    mask = ~(sources == targets)

    return adj[mask]
