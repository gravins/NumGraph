import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional, Tuple
from numgraph.utils import to_undirected

def _clique(num_nodes: int,
            weighted: bool = False,
            rng: Optional[Generator] = None) -> NDArray:
    
    adj_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    
    weights = None
    if weighted:
        if rng is None:
            rng = default_rng()
        weights = to_undirected(rng.uniform(low=0.0, high=1.0, size=(num_nodes, num_nodes)))

    return adj_matrix, weights


def clique_full(num_nodes: int, 
                weighted: bool = False,
                rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a complete graph, a.k.a. a clique.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    weighted : bool, optional
        If set to :obj:`True`, will return a dense representation of the weighted graph, by default :obj:`False`
    rng : Optional[Generator], optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    NDArray
        The clique in matrix representation :obj:`(num_nodes x num_nodes)`
    """
    adj_matrix, weights = _clique(num_nodes=num_nodes, weighted=weighted, rng=rng)

    adj_matrix = adj_matrix.astype(dtype=np.float32)

    return adj_matrix * weights if weighted else adj_matrix


def clique_coo(num_nodes: int, 
               weighted: bool = False,
               rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:
    """
     Returns a complete graph, a.k.a. a clique.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    weighted : bool, optional
        If set to :obj:`True`, will return a dense representation of the weighted graph, by default :obj:`False`
    rng : Optional[Generator], optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    NDArray
        The clique in COO representation :obj:`(num_edges x 2)`
    Optional[NDArray]
        Weights of the random graph.
    """

    adj_matrix, weights = _clique(num_nodes=num_nodes, weighted=weighted, rng=rng)
    if weighted:
        weights *= adj_matrix

    coo_matrix = np.argwhere(adj_matrix)
    coo_weights = np.expand_dims(weights[weights.nonzero()], -1) if weights is not None else None

    return coo_matrix, coo_weights
