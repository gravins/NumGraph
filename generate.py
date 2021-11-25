import numpy as np
from utils import remove_self_loops


def erdos_renyi_graph(num_nodes, prob, directed=False, rng=None):
    """
    Returns a random graph, also known as an Erdos-Renyi graph or a binomial graph.
    The model chooses each of the possible edges with a defined probability.

    Parameters
    ----------
    num_nodes : Int
        The number of nodes
    prob : Float
        Probability of an edge
    directed : Bool, optional
        If set to True, will return a directed graph, by default False
    rng : np.random.Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    np.ndarray
        The random graph (num_edges x 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = rng.random((num_nodes, num_nodes)) <= prob

    if not directed:
        mask = mask + mask.T

    mask = remove_self_loops(mask)

    return np.argwhere(mask)


def stochastic_block_model(block_sizes, probs, generator, rng=None):
    """
    Returns a stochastic block model graph.
    This model partitions the nodes into blocks of defined sizes,
    and places edges between pairs of nodes depending on a probability matrix.
    Such a matrix specifies edge probabilities between and inside blocks.

    Parameters
    ----------
    block_sizes : List[Int]
        Sizes of blocks
    probs : List[List[Float]]
        The squared probability matrix (num_blocks x num_blocks).
        The element i,j represents the edge probability between blocks i and j.
        The element i,i define the edge probability inside block i.
    generator : Callable
        A callable that generates communities with size depending on block_sizes
    rng : np.random.Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    np.ndarray
        The stochastic block model graph (num_edges x 2)
    """
    if rng is None:
        rng = np.random.default_rng()

    communities = [generator(b, probs[i], rng=rng) for i, b in enumerate(block_sizes)]

    # Update communities's indices
    sizes = {}
    first_id = 0
    for i in range(len(block_sizes)):
        communities[i] += first_id
        sizes[i] = first_id
        first_id += block_sizes[i]

    # Compute iter-block links
    edges = []
    for i in range(len(probs)):
        for j in range(len(probs)):
            if i == j: continue

            p = probs[i][j]
            size_c1, size_c2 = block_sizes[i], block_sizes[j]

            mask = rng.random((size_c1, size_c2)) <= p

            inter_block_edges = np.argwhere(mask)
            inter_block_edges[:, 0] += sizes[i]
            inter_block_edges[:, 1] += sizes[j]

            edges.append(inter_block_edges)

    return np.concatenate(edges + communities)


def full_grid(height, width):
    """
    Returns a full two-dimensional rectangular grid lattice graph.
    Example
    1 - 2 - 3
    | X | X |
    4 - 5 - 6

    Parameters
    ----------
    height : Int
        Number of vertices in the vertical axis
    width : Int
        Number of vertices in the horizontal axis

    Returns
    -------
    np.ndarray
        The full two-dimensional rectangular grid lattice graph (num_edges x 2)
    """
    w = width
    kernel = np.array([-w - 1, -1, w - 1, -w, w, -w + 1, 1, w + 1])
    return grid(height, width, kernel)


def simple_grid(height, width):
    """
    Returns a two-dimensional rectangular grid lattice graph.
    Example
    1 -- 2 -- 3
    |    |    |
    4 -- 5 -- 6

    Parameters
    ----------
    height : Int
        Number of vertices in the vertical axis
    width : Int
        Number of vertices in the horizontal axis

    Returns
    -------
    np.ndarray
        The two-dimensional rectangular grid lattice graph (num_edges x 2)
    """
    w = width
    kernel = np.array([-1, -w, w, 1])
    return grid(height, width, kernel)


def grid(height, width, kernel):
    """
    Ausiliar function for grid graph generation.

    Parameters
    ----------
    height : Int
        Number of vertices in the vertical axis
    width : Int
        Number of vertices in the horizontal axis
    kernel : List
        The kernel

    Returns
    -------
    np.ndarray
        The two-dimensional grid lattice graph (num_edges x 2)
    """
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
