from typing import Tuple, Callable, Optional, List, Union
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
import numpy as np

def _heat_graph_diffusion(generator: Callable,
                          timestamps: List, 
                          init_temp: Optional[Union[float, NDArray]] = None, 
                          num_nodes: Optional[int] = None,
                          return_coo: float = True,
                          rng: Optional[Generator] = None) -> Tuple[List[NDArray], List[NDArray]]:
    
    assert init_temp is None or (isinstance(init_temp, float) and init_temp > 0), f'init_temp can be None or float > 0, not {type(init_temp)} with value {init_temp}' 
    
    if rng is None:
        rng = default_rng()

    # Generate the graph
    edges = generator(rng) 
    assert isinstance(edges, tuple) and edges[0].shape[1] == 2, 'The generator must return a graph in COO representation.'
    edges, weights = edges

    if num_nodes is None:
        num_nodes = edges.max() + 1

    if init_temp is None:
        x = rng.uniform(low=0.0, high=0.2, size=(num_nodes, 1))
    elif isinstance(init_temp, float):
        x = np.full((num_nodes,1), init_temp)
    else:
        x = init_temp

    # Compute the Laplacian matrix
    adj_mat = np.zeros((num_nodes, num_nodes))
    adj_mat[edges[:, 0], edges[:, 1]] = 1 if weights is None else weights
    self_loops = np.diag(adj_mat)
    np.fill_diagonal(adj_mat, 0)
    degree = np.diag(np.sum(adj_mat, axis=1))
    new_degree = np.linalg.inv(np.sqrt(degree))
    L = np.eye(num_nodes) - new_degree @ adj_mat @ new_degree # Normalized laplacian

    eigenvalues, eigenvectors = np.linalg.eig(L)
    Lambda = np.diag(eigenvalues)
    l = np.zeros_like(Lambda)
    
    xs = []
    for t in timestamps:
        # Closed form solution of the Graph Heat Diffusion equation (ie, e^{-tL}x_0)
        np.fill_diagonal(l, np.exp(-t * np.diag(Lambda)))
        xs.append(eigenvectors @ l @ np.linalg.inv(eigenvectors) @ x)
        
    if return_coo:
        return [(edges, weights)] * len(timestamps), xs
    else:
        adj_mat += self_loops 
        return [adj_mat] * len(timestamps), xs


def heat_graph_diffusion_coo(generator: Callable,
                             timestamps: List, 
                             init_temp: Optional[Union[float, NDArray]] = None, 
                             num_nodes: Optional[int] = None,
                             rng: Optional[Generator] = None) -> Tuple[List[Tuple[NDArray, NDArray]], List[NDArray]]:
    """
    Returns heat diffusion over a graph computed with the closed form solution. 
    The model simulates the diffusion of heat on a given graph through the graph heat equation. 
    Each node is characterized by the temperature. The process is evaluated on the predefined 
    :obj:`timestamps`. The simulation graph has fixed nodes and edges along the temporal axis.

    Parameters
    ----------
    generator : Callable
        A callable that takes as input a rng and generates the simulation graph
    timestamps: List, 
        The list of timestamps in which the diffusion is evaluated
    init_temp : Union[float, NDArray], optional
        The initial temperature of the nodes. If :obj:`None` it computes a random temperature between :obj:`0.` and :obj:`0.2`, by default :obj:`None`
    num_nodes : int, optional
        The number of nodes in the simulation graph, by default :obj:`None`
    rng : Generator, optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    List[Tuple[NDArray, NDArray]]
        The list of graph's snapshots :obj:`(T x 2)`: each snapshot is a tuple containing the graph 
        in COO representation :obj:`(snapshot_num_edges x 2)` and the weights :obj:`(snapshot_num_edges x 1)`
    List[NDArray]
        The list of nodes' states :obj:`(T x (snapshot_num_nodes, ))`
    """
    return _heat_graph_diffusion(generator = generator,
                                 timestamps = timestamps, 
                                 init_temp = init_temp, 
                                 num_nodes = num_nodes,
                                 return_coo = True,
                                 rng = rng)


def heat_graph_diffusion_full(generator: Callable,
                              timestamps: List,
                              init_temp: Optional[Union[float, NDArray]] = None, 
                              num_nodes: Optional[int] = None,
                              rng: Optional[Generator] = None) -> Tuple[List[NDArray], List[NDArray]]:
    """
    
    Returns heat diffusion over a graph computed with the closed form solution. 
    The model simulates the diffusion of heat on a given graph through the graph heat equation. 
    Each node is characterized by the temperature. The process is evaluated on the predefined 
    :obj:`timestamps`. The simulation graph has fixed nodes and edges along the temporal axis.

    Parameters
    ----------
    generator : Callable
        A callable that takes as input a rng and generates the simulation graph
    timestamps: List, 
        The list of timestamps in which the diffusion is evaluated
    init_temp : Union[float, NDArray], optional
        The initial temperature of the nodes. If :obj:`None` it computes a random temperature between :obj:`0.` and :obj:`0.2`, by default :obj:`None`
    num_nodes : int, optional
        The number of nodes in the simulation graph, by default :obj:`None`
    rng : Generator, optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    List[NDArray]
        the list of graph's snapshots in matrix representation :obj:`(T x (num_nodes x num_nodes))`
    List[NDArray]
        the list of nodes' states :obj:`(T x (num_nodes, ))`
    """
    return _heat_graph_diffusion(generator = generator, 
                                 timestamps = timestamps, 
                                 init_temp = init_temp, 
                                 num_nodes = num_nodes,
                                 return_coo = False,
                                 rng = rng)
 
