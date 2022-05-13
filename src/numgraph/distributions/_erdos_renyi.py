import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional, Tuple
from numgraph.utils import to_undirected, remove_self_loops


def _erdos_renyi(num_nodes: int,
                 prob: float,
                 directed: bool = False,
                 weighted: bool = False,
                 rng: Optional[Generator] = None) -> NDArray:

    assert num_nodes >= 0 and 0 < prob <= 1

    if rng is None:
        rng = default_rng()

    adj_matrix = rng.random((num_nodes, num_nodes)) <= prob
    adj_matrix = remove_self_loops(adj_matrix)

    if not directed:
        adj_matrix = adj_matrix + adj_matrix.T

    weights = None
    if weighted:
        weights = rng.uniform(low=0.0, high=1.0, size=(num_nodes, num_nodes))
        if not directed:
            weights = to_undirected(weights)

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
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The random graph in COO representation (num_edges x 2).
    Optional[NDArray]
        Weights of the random graph.
    """
    adj_matrix, weights = _erdos_renyi(num_nodes=num_nodes,
                                       prob=prob,
                                       directed=directed,
                                       weighted=weighted,
                                       rng=rng)

    if weighted:
        weights *= adj_matrix

    coo_matrix = np.argwhere(adj_matrix)
    coo_weights = np.expand_dims(weights[weights.nonzero()], -1) if weights is not None else None

    return coo_matrix, coo_weights


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
        If set to True, will return a dense representation of the weighted graph, by default False
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
