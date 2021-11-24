from numpy.typing import NDArray

def to_dense(adj: NDArray, num_nodes=None):
    if not num_nodes:
        num_nodes = np.max(adj) + 1

    dense_adj = np.zeros((num_nodes, num_nodes))

    for i, j in adj:
        dense_adj[i, j] = 1

    return dense_adj

def dense(generator):
    return lambda *args: to_dense(generator(*args))

def remove_self_loops(adj: NDArray):

    row, col = adj.shape

    if row == col:
        return adj * (1 - np.eye(row, col, dtype=np.bool8))

    raise NotImplementedError()
