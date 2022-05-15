from typing import Tuple, Optional
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


def to_undirected(adj: NDArray) -> NDArray:
    """
    Turns a directed edge_list into a non-directed one

    Parameters
    ----------
    edge_list : NDArray
        A directed adjacency matrix (num_nodes x num_nodes), or edge list (num_edges x 2)

    Returns
    -------
    NDArray
        An undirected adjacency matrix (num_nodes x num_nodes), or the edge list ((2*num_edges) x 2)
    """
    row, col = adj.shape

    if row == col:
        # Case of a squared dense adj matrix
        return np.triu(adj) + np.triu(adj, 1).T

    sources, targets = adj[:, 0], adj[:, 1]
    sources, targets = sources.reshape((-1, 1)), targets.reshape((-1, 1))

    new_edges = np.concatenate((targets, sources), axis=1)
    adj = np.concatenate((adj, new_edges), axis=0)

    return adj


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


def unsorted_coalesce(edge_list: NDArray, weights: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
    """
    Polishes an edge list by removing duplicates and by sorting the edges

    Parameters
    ----------
    edge_list : NDArray
        An edge list (num_edges x 2)
    weights : NDArray
        The weights (num_edges x 1)
    Returns
    -------
    NDArray
        An unsorted edge list with no duplicated edges (new_num_edges x 2)
    NDArray
        The unsorted weigths associated to the new edge list (new_num_edges x 1)
    """
    indexes = sorted(np.unique(edge_list, return_index=True, axis=0)[1])
    return edge_list[indexes], weights[indexes] if weights is not None else weights


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


def remove_self_loops(adj: NDArray) -> NDArray:
    """
    Removes every self-loop in the graph given by adj

    Parameters
    ----------
    adj : NDArray
        The adjancency matrix (num_nodes x num_nodes), or the edge_list (num_edges x 2)

    Returns
    -------
    NDAarray
        The adjacency matrix (num_nodes x num_nodes), or the list of edges (new_num_edges x 2), 
        without self-loops.

    Raises
    ------
    NotImplementedError
    """
    row, col = adj.shape

    if row == col:
        # Case of a squared dense adj matrix
        np.fill_diagonal(adj, 0)
        return adj

    sources, targets = adj[:, 0], adj[:, 1]
    mask = ~(sources == targets)

    return adj[mask]
