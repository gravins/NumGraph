import numpy as np


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


def dense(generator):
    """
    Compute the squared adjacency matrix of a generated graph

    Parameters
    ----------
    generator : Callable
        A callable that generates the graphs

    Returns
    -------
    np.ndarray
        The squared adjacency matrix (num_nodes x num_nodes)
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

    raise NotImplementedError()
