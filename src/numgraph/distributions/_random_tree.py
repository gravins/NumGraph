import numpy as np
from collections import Counter
from numpy.typing import NDArray
from typing import Optional, Tuple
from numpy.random import Generator, default_rng
from numgraph.utils import to_undirected


def _random_tree(num_nodes: int, 
                 directed: bool = True, 
                 weighted: bool = False,
                 rng: Optional[Generator] = None) -> NDArray:

    if rng is None:
        rng = default_rng()

    prufer_seq = [rng.choice(range(num_nodes)) for _ in range(num_nodes - 2)]

    # Node degree is equivalent to the number of times it appears in the sequence + 1
    degree = Counter(prufer_seq + list(range(num_nodes)))

    edges = []
    visited = set()
    for v in prufer_seq:
        for u in range(num_nodes):
            if degree[u] == 1:
                edges.append([v, u])
                degree[v] -= 1
                degree[u] -= 1
                visited.add(u)
                break

    u, v = degree.keys() - visited
    edges.append([u,v])
    edges = np.asarray(edges)

    weights = None
    if weighted:
        if rng is None:
            rng = default_rng()
        weights = rng.uniform(low=0.0, high=1.0, size=(len(edges), 1))

    if not directed:
        edges = to_undirected(edges)
        weights = np.vstack((weights, weights)) if weights is not None else None

    return edges, weights


def random_tree_coo(num_nodes: int,
                    directed: bool = True, 
                    weighted: bool = False,
                    rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:
    """
    Returns a random tree computed using a random Prufer sequence.

    Parameters
    ----------
    num_nodes : int
        [description]
    directed : bool, optional
        If set to True, will return a directed graph, by default True
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The random tree in COO representation (num_edges x 2)
    Optional[NDArray]
        Weights of the random graph.
    """

    return _random_tree(num_nodes = num_nodes,
                        directed = directed,
                        weighted = weighted,
                        rng = rng)
    

def random_tree_full(num_nodes: int,
                     directed: bool = True, 
                     weighted: bool = False,
                     rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a random tree computed using a random Prufer sequence.

    Parameters
    ----------
    num_nodes : int
        [description]
    directed : bool, optional
        If set to True, will return a directed graph, by default True
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The random tree in matrix representation (num_edges x 2)
    """

    coo_matrix, weights = _random_tree(num_nodes = num_nodes,
                                       directed = directed,
                                       weighted = weighted,
                                       rng = rng)

    # Fill adj_matrix with the weights
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[coo_matrix[:, 0], coo_matrix[:, 1]] = np.squeeze(weights) if weighted else 1
    
    return adj_matrix