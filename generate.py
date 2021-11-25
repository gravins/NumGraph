import numpy as np
from utils import remove_self_loops

def erdos_renyi(num_nodes, prob, directed=False, rng=None):
    r"""Returns the :obj:`edge_index` of a random Erdos-Renyi graph.

    Args:
        num_nodes (int): The number of nodes.
        edge_prob (float): Probability of an edge.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)
    """

    if rng is None:
        rng = np.random.default_rng()

    mask = rng.random((num_nodes, num_nodes)) <= prob

    if not directed:
        mask = mask + mask.T

    mask = remove_self_loops(mask)

    return np.argwhere(mask)

def full_grid(height, width):
    w = width
    kernel = np.array([-w - 1, -1, w - 1, -w, w, -w + 1, 1, w + 1])
    return grid(height, width, kernel)

def simple_grid(height, width):
    w = width
    kernel = np.array([-1, -w, w, 1])
    return grid(height, width, kernel)

def grid(height, width, kernel):
    num_nodes = height * width
    K = len(kernel)

    sources = np.arange(num_nodes, dtype=np.long).repeat(K)
    targets = sources + np.tile(kernel, num_nodes)
    mask = (targets >= 0) & (targets < num_nodes)

    sources, targets = sources[mask].reshape((-1, 1)), targets[mask].reshape((-1, 1))
    edges = np.concatenate([sources, targets], axis=1)

    bounds = np.arange(width-1, num_nodes-1, width, dtype=np.long)
    bounds = bounds.reshape((-1, 1))
    bounds = np.concatenate(
        [bounds, bounds + 1],
        axis=1
    )

    # Remove edges (u,v) from a boundary node to the first node of the new row.
    submask_1 = ((edges[:, 0] + 1) % width == 0) & ((edges[:, 1]) % width == 0)
    # As the graph is undirected, remove the corresponding edges (v, u).
    submask_2 = ((edges[:, 0]) % width == 0) & ((edges[:, 1] + 1) % width == 0)

    mask = ~(submask_1 | submask_2)

    return edges[mask]
