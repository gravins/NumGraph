import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional, Tuple
from numgraph.utils import remove_self_loops, to_undirected


def _barabasi_albert(num_nodes: int, 
                     num_edges: int, 
                     weighted: bool = False,
                     rng: Optional[Generator] = None) -> NDArray:

    assert num_nodes >= 0 and num_edges > 0 and num_edges < num_nodes

    if rng is None:
        rng = default_rng()

    sources, targets = np.arange(num_edges), rng.permutation(num_edges)

    for i in range(num_edges, num_nodes):
        sources = np.concatenate([sources, np.full((num_edges, ), i, dtype=np.int64)])
        choice = rng.choice(np.concatenate([sources, targets]), num_edges)
        targets = np.concatenate([targets, choice])

    sources, targets = sources.reshape((-1, 1)), targets.reshape((-1, 1))
    edge_list = np.concatenate([sources, targets], axis=1)

    edge_list = remove_self_loops(edge_list)
    edge_list = to_undirected(edge_list)

    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[edge_list[:, 0], edge_list[:, 1]] = 1
    
    weights = None
    if weighted:
        weights = rng.uniform(low=0.0, high=1.0, size=(num_nodes, num_nodes))
        weights = to_undirected(weights)

    return adj_matrix, weights



def barabasi_albert_coo(num_nodes: int, 
                        num_edges: int, 
                        weighted: bool = False,
                        rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:
    """
    Returns a graph sampled from the Barabasi-Albert (BA) model. The graph is built
    incrementally by adding :obj:`num_edges` arcs from a new node to already existing ones with
    preferential attachment towards nodes with high degree.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    num_edges : int
        The number of edges
    weighted : bool, optional
        If set to :obj:`True`, will return a dense representation of the weighted graph, by default :obj:`False`
    rng : Optional[Generator], optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    NDArray
        The Barabasi-Albert graph in COO representation :obj:`(num_edges x 2)`
    Optional[NDArray]
        Weights of the random graph.
    """

    adj_matrix, weights = _barabasi_albert(num_nodes=num_nodes, 
                                           num_edges=num_edges, 
                                           weighted=weighted,
                                           rng=rng)
    
    if weighted:
        weights *= adj_matrix

    coo_matrix = np.argwhere(adj_matrix)
    coo_weights = np.expand_dims(weights[weights.nonzero()], -1) if weights is not None else None

    return coo_matrix, coo_weights


def barabasi_albert_full(num_nodes: int, 
                         num_edges: int, 
                         weighted: bool = False,
                         rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a graph sampled from the Barabasi-Albert (BA) model. The graph is built
    incrementally by adding `num_edges` arcs from a new node to already existing ones with
    preferential attachment towards nodes with high degree.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    num_edges : int
        The number of edges
    weighted : bool, optional
        If set to :obj:`True`, will return a dense representation of the weighted graph, by default :obj:`False`
    rng : Optional[Generator], optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    NDArray
        The Barabasi-Albert graph in matrix representation :obj:`(num_nodes x num_nodes)`
    """

    adj_matrix, weights = _barabasi_albert(num_nodes=num_nodes, 
                                           num_edges=num_edges, 
                                           weighted=weighted,
                                           rng=rng)

    adj_matrix = adj_matrix.astype(dtype=np.float32)

    return adj_matrix * weights if weighted else adj_matrix