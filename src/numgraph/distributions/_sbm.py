import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from typing import Optional, Tuple, List, Callable
from numgraph.utils import coalesce, unsorted_coalesce


def _stochastic_block_model(block_sizes: List[int], 
                            probs: List[List[float]],
                            generator: Callable, 
                            directed: bool = False,
                            weighted: bool = False,
                            rng: Optional[Generator] = None) -> NDArray:
    
    assert all(block_sizes) and len(probs) == len(block_sizes)
    assert all([len(p) == len(probs) for p in probs]) and all([all(p) for p in probs])

    if rng is None:
        rng = default_rng()

    communities = []
    for i, b in enumerate(block_sizes):
        edges = generator(b, probs[i][i], rng)
        assert isinstance(edges, tuple) and edges[0].shape[1] == 2, 'The generator must return a graph in COO representation.'
        communities.append(edges[0])

    # Update communities's indices
    sizes = {}
    first_id = 0
    for i in range(len(block_sizes)):
        communities[i] += first_id
        sizes[i] = first_id
        first_id += block_sizes[i]

    # Compute inter-block links
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

    edges = np.concatenate(edges + communities)
    if not directed:
        edges = np.vstack((edges, edges[:, [1,0]]))
    edges = coalesce(edges)

    weights = None
    if weighted:
        weights = rng.uniform(low=0.0, high=1.0, size=(sum(block_sizes), sum(block_sizes)))
        if not directed:
            weights = np.tril(weights) + np.triu(weights.T, 1)
        weights = np.expand_dims(weights[edges[:,0], edges[:,1]], axis=-1)

    return edges, weights



def stochastic_block_model_coo(block_sizes: List[int], 
                               probs: List[List[float]],
                               generator: Callable,
                               directed: bool = False,
                               weighted: bool = False,
                               rng: Optional[Generator] = None) -> Tuple[NDArray, Optional[NDArray]]:
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
        The squared probability matrix :obj:`(num_blocks x num_blocks)`.
        The element :obj:`i,j` represents the edge probability between blocks :obj:`i` and :obj:`j`.
        The element :obj:`i,i` define the edge probability inside block :obj:`i`.
    generator : Callable
        A callable that generates communities with size depending on block_sizes
    directed : bool, optional
        If set to :obj:`True`, will return a directed graph, by default :obj:`False`
    weighted : bool, optional
        If set to :obj:`True`, will return a dense representation of the weighted graph, by default :obj:`False`
    rng : Generator, optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    NDArray
        The stochastic block model graph in COO representation :obj:`(num_edges x 2)`.
    Optional[NDArray]
        Weights of the random graph.

    Note
    ----
        The generator takes as input the block size, the probability, and the rng
    """

    coo_matrix, coo_weights = _stochastic_block_model(block_sizes=block_sizes,
                                                      probs=probs,
                                                      generator=generator,
                                                      directed=directed,
                                                      weighted=weighted,
                                                      rng=rng)

    return coo_matrix, coo_weights


def stochastic_block_model_full(block_sizes: List[int], 
                                probs: List[List[float]],
                                generator: Callable,
                                directed: bool = False,
                                weighted: bool = False,
                                rng: Optional[Generator] = None) -> NDArray:
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
        The squared probability matrix :obj:`(num_blocks x num_blocks)`.
        The element :obj:`i,j` represents the edge probability between blocks :obj:`i` and :obj:`j`.
        The element :obj:`i,i` define the edge probability inside block :obj:`i`.
    generator : Callable
        A callable that generates communities with size depending on block_sizes
    directed : bool, optional
        If set to :obj:`True`, will return a directed graph, by default :obj:`False`
    weighted : bool, optional
        If set to :obj:`True`, will return a dense representation of the weighted graph, by default :obj:`False`
    rng : Generator, optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    NDArray
        The stochastic block model graph in matrix representation :obj:`(num_nodes x num_nodes)`.

    Note
    ----
        The generator takes as input the block size, the probability, and the rng
    """

    coo_matrix, weights = _stochastic_block_model(block_sizes=block_sizes,
                                                  probs=probs,
                                                  generator=generator,
                                                  directed=directed,
                                                  weighted=weighted,
                                                  rng=rng)
    # Fill adj_matrix with the weights
    num_nodes = sum(block_sizes)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[coo_matrix[:, 0], coo_matrix[:, 1]] = np.squeeze(weights) if weighted else 1

    return adj_matrix