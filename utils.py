import numpy as np
from numpy.typing import NDArray


def to_dense(adj, num_nodes=None):
    """
    Converts a list of edges in a squared adjacency matrix

    Parameters
    ----------
    adj : np.ndarray
        The list of edges (num_edges x 2)
    num_nodes : int, optional
        The number of nodes in the graph, by default None

    Returns
    -------
    np.ndarray
        The squared adjacency matrix (num_nodes x num_nodes)
    """
    if not num_nodes:
        num_nodes = np.max(adj) + 1

    dense_adj = np.zeros((num_nodes, num_nodes))

    for i, j in adj:
        dense_adj[i, j] = 1

    return dense_adj

def to_undirected(edge_list: NDArray):
    """
    Turns a directed edge_list into a non-directed one

    Parameters
    ----------
    edge_list : NDArray
        A directed edge list

    Returns
    -------
    np.ndarray
        An undirected edge list
    """
    sources, targets = edge_list[:, 0], edge_list[:, 1]
    sources, targets = sources.reshape((-1, 1)), targets.reshape((-1, 1))

    new_edges = np.concatenate((targets, sources), axis=1)
    edge_list = np.concatenate((edge_list, new_edges), axis=0)

    return edge_list

def coalesce(edge_list: NDArray):
    """
    Polishes an edge list by removing duplicates and by sorting the edges

    Parameters
    ----------
    edge_list : NDArray
        An edge list

    Returns
    -------
    edge_list
        A sorted edge list with no duplicated edges
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
    return lambda *args: to_dense(generator(*args))


def remove_self_loops(adj):
    """
    Removes every self-loop in the graph given by adj

    Parameters
    ----------
    adj : np.ndarray
        The adjancency matrix

    Returns
    -------
    np.ndarray
        The list of edges without self-loops (num_edges x 2)

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
