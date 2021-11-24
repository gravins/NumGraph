import numpy as np
from utils import remove_self_loops

def erdos_renyi_graph(num_nodes, prob, directed=False, rng=None):
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
