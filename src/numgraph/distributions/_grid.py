import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional, Tuple
from numgraph.utils import to_undirected


def _grid(height: int, 
          width: int, 
          kernel: NDArray,
          directed: bool = False,
          weighted: bool = False, 
          rng: Optional[Generator] = None) -> NDArray:

    """
    Ausiliar function for grid graph generation.
    """
    num_nodes = height * width
    K = len(kernel)

    sources = np.arange(num_nodes, dtype=np.int64).repeat(K)
    targets = sources + np.tile(kernel, num_nodes)
    mask = (targets >= 0) & (targets < num_nodes)

    sources, targets = sources[mask].reshape((-1, 1)), targets[mask].reshape((-1, 1))
    edge_list = np.concatenate([sources, targets], axis=1)

    # Remove edges (u,v) from a boundary node to the first node of the new row.
    submask_1 = ((edge_list[:, 0] + 1) % width == 0) & ((edge_list[:, 1]) % width == 0)
    # As the graph is undirected, remove the corresponding edges (v, u).
    submask_2 = ((edge_list[:, 0]) % width == 0) & ((edge_list[:, 1] + 1) % width == 0)

    mask = ~(submask_1 | submask_2)

    edge_list = edge_list[mask]
    num_nodes = height * width
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[edge_list[:,0], edge_list[:,1]] = 1

    if not directed:
        adj_matrix = adj_matrix + adj_matrix.T
        adj_matrix[adj_matrix.nonzero()] = 1

    weights = None
    if weighted:
        if rng is None:
            rng = default_rng()
        weights = rng.uniform(low=0.0, high=1.0, size=(num_nodes, num_nodes))
        weights = to_undirected(weights)

    return adj_matrix, weights


def grid_full(height: int, 
              width: int, 
              directed: bool=False,
              weighted: bool = False, 
              rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a full undirected two-dimensional rectangular grid lattice graph.
    Example
    1 - 2 - 3
    | X | X |
    4 - 5 - 6

    Parameters
    ----------
    height : int
        Number of vertices in the vertical axis
    width : int
        Number of vertices in the horizontal axis
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The full undirected two-dimensional rectangular grid lattice graph in matrix representation (num_nodes x num_nodes)
    """
    w = width
    kernel = np.array([-w - 1, -w, -w + 1, -1, w, w - 1, w, w + 1])
    adj_matrix, weights = _grid(height, width, kernel, directed, weighted, rng)

    adj_matrix = adj_matrix.astype(dtype=np.float32)

    return adj_matrix * weights if weighted else adj_matrix


def grid_coo(height: int, 
             width: int, 
             directed: bool=False,
             weighted: bool = False, 
             rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:
    """
    Returns a full undirected two-dimensional rectangular grid lattice graph.
    Example
    1 - 2 - 3
    | X | X |
    4 - 5 - 6

    Parameters
    ----------
    height : int
        Number of vertices in the vertical axis
    width : int
        Number of vertices in the horizontal axis
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The full undirected two-dimensional rectangular grid lattice graph in COO representation (num_edges x 2)
    Optional[NDArray]
        Weights of the random graph.
    """
    w = width
    kernel = np.array([-w - 1, -w, -w + 1, -1, w, w - 1, w, w + 1])
    adj_matrix, weights = _grid(height, width, kernel, directed, weighted, rng)

    if weighted:
        weights *= adj_matrix

    coo_matrix = np.argwhere(adj_matrix)
    coo_weights = np.expand_dims(weights[weights.nonzero()], -1) if weights is not None else None

    return coo_matrix, coo_weights


def simple_grid_full(height: int, 
                     width: int,
                     directed: bool=False,
                     weighted: bool = False, 
                     rng: Optional[Generator] = None) -> NDArray:
    """
    Returns an undirected two-dimensional rectangular grid lattice graph.
    Example
    1 -- 2 -- 3
    |    |    |
    4 -- 5 -- 6

    Parameters
    ----------
    height : int
        Number of vertices in the vertical axis
    width : int
        Number of vertices in the horizontal axis
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The undirected two-dimensional rectangular grid lattice graph in matrix representation (num_nodes x num_nodes)
    """
    w = width
    kernel = np.array([-w, -1, 1, w])
    adj_matrix, weights = _grid(height, width, kernel, directed, weighted, rng)

    adj_matrix = adj_matrix.astype(dtype=np.float32)

    return adj_matrix * weights if weighted else adj_matrix


def simple_grid_coo(height: int, 
                    width: int,
                    directed: bool=False,
                    weighted: bool = False, 
                    rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:
    """
    Returns an undirected two-dimensional rectangular grid lattice graph.
    Example
    1 -- 2 -- 3
    |    |    |
    4 -- 5 -- 6

    Parameters
    ----------
    height : int
        Number of vertices in the vertical axis
    width : int
        Number of vertices in the horizontal axis
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    weighted : bool, optional
        If set to True, will return a dense representation of the weighted graph, by default False
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The undirected two-dimensional rectangular grid lattice graph in COO representation (num_edges x 2)
    Optional[NDArray]
        Weights of the random graph.
    """
    w = width
    kernel = np.array([-w, -1, 1, w])
    adj_matrix, weights = _grid(height, width, kernel, directed, weighted, rng)

    if weighted:
        weights *= adj_matrix

    coo_matrix = np.argwhere(adj_matrix)
    coo_weights = np.expand_dims(weights[weights.nonzero()], -1) if weights is not None else None

    return coo_matrix, coo_weights