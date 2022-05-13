import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional, Tuple
from numgraph.utils import to_undirected

def _star(num_nodes: int,
          directed: bool = False,
          weighted: bool = False,
          rng: Optional[Generator] = None) -> NDArray:
    
    edges = np.stack([np.zeros((num_nodes - 1, ), dtype = int), np.arange(1, num_nodes)],
                      axis=1)
    
    weights = None
    if weighted:
        if rng is None:
            rng = default_rng()
        weights = rng.uniform(low=0.0, high=1.0, size=(len(edges), 1))

    if not directed:
        edges = to_undirected(edges)
        weights = np.vstack((weights, weights)) if weights is not None else None

    return edges, weights


def star_full(num_nodes: int, 
              directed: bool = False,
              weighted: bool = False,
              rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a star graph.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The star graph in matrix representaion (num_nodes x num_nodes)
    """
    coo_matrix, weights = _star(num_nodes=num_nodes, directed=directed, weighted=weighted, rng=rng)

    # Fill adj_matrix with the weights
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[coo_matrix[:, 0], coo_matrix[:, 1]] = np.squeeze(weights) if weighted else 1
    
    return adj_matrix


def star_coo(num_nodes: int, 
             directed: bool = False,
             weighted: bool = False,
             rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:
    """
    Returns a star graph.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The star graph in COO representation (num_edges x 2)
    Optional[NDArray]
        Weights of the random graph.
    """
    return _star(num_nodes=num_nodes, directed=directed, weighted=weighted, rng=rng)
