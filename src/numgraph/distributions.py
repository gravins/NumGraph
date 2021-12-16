import numpy as np
from collections import Counter
from numpy.typing import NDArray
from itertools import combinations
from typing import List, Callable, Optional
from numpy.random import Generator, default_rng
from numgraph.utils import remove_self_loops, to_undirected, coalesce


def erdos_renyi(num_nodes: int, prob: float, directed: bool = False,
                rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a random graph, also known as an Erdos-Renyi graph or a binomial graph.
    The model chooses each of the possible edges with a defined probability.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    prob : float
        Probability of an edge
    directed : bool, optional
        If set to True, will return a directed graph, by default False
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The random graph (num_edges x 2)
    """
    assert num_nodes >= 0 and prob <=1 and prob > 0

    if rng is None:
        rng = default_rng()

    mask = rng.random((num_nodes, num_nodes)) <= prob

    if not directed:
        mask = mask + mask.T

    mask = remove_self_loops(mask)

    edge_list = np.argwhere(mask)
    return edge_list


def barabasi_albert(num_nodes: int, num_edges: int, rng: Optional[Generator] = None) -> NDArray:
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
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The Barabasi-Albert (num_edges x 2)
    """
    assert num_nodes >= 0 and num_edges > 0 and num_edges < num_nodes

    if rng is None:
        rng = default_rng()

    sources, targets = np.arange(num_edges), rng.permutation(num_edges)

    for i in range(num_edges, num_nodes):
        sources = np.concatenate([sources, np.full((num_edges, ), i, dtype=np.long)])
        choice = rng.choice(np.concatenate([sources, targets]), num_edges)
        targets = np.concatenate([targets, choice])

    sources, targets = sources.reshape((-1, 1)), targets.reshape((-1, 1))
    edge_list = np.concatenate([sources, targets], axis=1)

    edge_list = remove_self_loops(edge_list)
    edge_list = to_undirected(edge_list)

    return coalesce(edge_list)


def clique(num_nodes: int) -> NDArray:
    """
    Returns a complete graph, a.k.a. a clique.

    Parameters
    ----------
    num_nodes : int
        The number of nodes

    Returns
    -------
    NDArray
        The clique (num_edges x 2)
    """
    edge_list = np.array(list(combinations(range(num_nodes), r=2)))
    edge_list = to_undirected(edge_list)
    return coalesce(edge_list)


def stochastic_block_model(block_sizes: List[int], probs: List[List[float]],
                           generator: Callable, rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a stochastic block model graph.
    This model partitions the nodes into blocks of defined sizes,
    and places edges between pairs of nodes depending on a probability matrix.
    Such a matrix specifies edge probabilities between and inside blocks.

    Parameters
    ----------
    block_sizes : List[int]
        Sizes of blocks
    probs : List[List[float]]
        The squared probability matrix (num_blocks x num_blocks).
        The element i,j represents the edge probability between blocks i and j.
        The element i,i define the edge probability inside block i.
    generator : Callable
        A callable that generates communities with size depending on block_sizes
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The stochastic block model graph (num_edges x 2)
    """
    assert all(block_sizes) and len(probs) == len(block_sizes)
    assert all([len(p) == len(probs) for p in probs]) and all([all(p) for p in probs])

    if rng is None:
        rng = default_rng()

    communities = [generator(b, probs[i][i], rng) for i, b in enumerate(block_sizes)]

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


def full_grid(height: int, width: int) -> NDArray:
    """
    Returns a full two-dimensional rectangular grid lattice graph.
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

    Returns
    -------
    NDArray
        The full two-dimensional rectangular grid lattice graph (num_edges x 2)
    """
    w = width
    kernel = np.array([-w - 1, -w, -w + 1, -1, w, w - 1, w, w + 1])
    return grid(height, width, kernel)


def simple_grid(height: int, width: int) -> NDArray:
    """
    Returns a two-dimensional rectangular grid lattice graph.
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

    Returns
    -------
    NDArray
        The two-dimensional rectangular grid lattice graph (num_edges x 2)
    """
    w = width
    kernel = np.array([-w, -1, 1, w])
    return grid(height, width, kernel)


def grid(height: int, width: int, kernel: NDArray) -> NDArray:
    """
    Ausiliar function for grid graph generation.

    Parameters
    ----------
    height : int
        Number of vertices in the vertical axis
    width : int
        Number of vertices in the horizontal axis
    kernel : NDArray
        The kernel

    Returns
    -------
    NDArray
        The two-dimensional grid lattice graph (num_edges x 2)
    """
    num_nodes = height * width
    K = len(kernel)

    sources = np.arange(num_nodes, dtype=np.long).repeat(K)
    targets = sources + np.tile(kernel, num_nodes)
    mask = (targets >= 0) & (targets < num_nodes)

    sources, targets = sources[mask].reshape((-1, 1)), targets[mask].reshape((-1, 1))
    edge_list = np.concatenate([sources, targets], axis=1)

    # Remove edges (u,v) from a boundary node to the first node of the new row.
    submask_1 = ((edge_list[:, 0] + 1) % width == 0) & ((edge_list[:, 1]) % width == 0)
    # As the graph is undirected, remove the corresponding edges (v, u).
    submask_2 = ((edge_list[:, 0]) % width == 0) & ((edge_list[:, 1] + 1) % width == 0)

    mask = ~(submask_1 | submask_2)

    return edge_list[mask]


def star(num_nodes: int, directed: bool = False) -> NDArray:
    """
    Returns a star graph.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    directed : bool, optional
        If set to True, will return a directed graph, by default False

    Returns
    -------
    NDArray
        The star graph (num_edges x 2)
    """
    edges = np.stack([np.zeros((num_nodes - 1, ), dtype = int), np.arange(1, num_nodes)],
                      axis=1)

    if not directed:
        edges = to_undirected(edges)

    return edges


def random_tree(num_nodes: int, directed: bool = True, rng: Optional[Generator] = None) -> NDArray:
    """
    Returns a random tree computed using a random Prufer sequence.

    Parameters
    ----------
    num_nodes : int
        [description]
    directed : bool, optional
        If set to True, will return a directed graph, by default True
    rng : Optional[Generator], optional
        Numpy random number generator, by default None

    Returns
    -------
    NDArray
        The random tree (num_edges x 2)
    """
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

    if not directed:
        edges = to_undirected(edges)

    return edges
