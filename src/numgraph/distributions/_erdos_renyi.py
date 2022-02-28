import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional, Tuple


def _erdos_renyi(num_nodes: int,
                 prob: float,
                 directed: bool = False,
                 weighted: bool = False,
                 rng: Optional[Generator] = None) -> NDArray:

    assert num_nodes >= 0 and 0 < prob <= 1

    if rng is None:
        rng = default_rng()

    adj_matrix = rng.random((num_nodes, num_nodes)) <= prob

    if not directed:
        adj_matrix = adj_matrix + adj_matrix.T

    weights = None
    if weighted:
        weights = rng.uniform(low=0.0, high=1.0, size=(num_nodes, num_nodes))
        if not directed:
            weights = np.triu(weights)
            weights = weights + weights.T

    return adj_matrix, weights


def erdos_renyi_coo(num_nodes: int,
                    prob: float,
                    directed: bool = False,
                    weighted: bool = False,
                    rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:

    """
    Returns a random graph, also known as an Erdos-Renyi graph or a binomial graph.
    The model chooses each of the possible edges with a defined probability.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    prob : float
        Probability of an edge
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The random graph in COO representation (num_edges x 2).
    """
    adj_matrix, weights = _erdos_renyi(num_nodes=num_nodes,
                                       prob=prob,
                                       directed=directed,
                                       weighted=weighted,
                                       rng=rng)

    coo_matrix = np.argwhere(adj_matrix)

    return coo_matrix


def erdos_renyi_full(num_nodes: int,
                     prob: float,
                     directed: bool = False,
                     weighted: bool = False,
                     rng: Optional[Generator] = None) -> NDArray:

    """
    Returns a random graph, also known as an Erdos-Renyi graph or a binomial graph.
    The model chooses each of the possible edges with a defined probability.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    prob : float
        Probability of an edge
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The random graph in matrix representation (num_nodes x num_nodes).
    """
    adj_matrix, weights = _erdos_renyi(num_nodes=num_nodes,
                                       prob=prob,
                                       directed=directed,
                                       weighted=weighted,
                                       rng=rng)

    adj_matrix = adj_matrix.astype(dtype=np.float32)

    return adj_matrix * weights if weighted else adj_matrix
