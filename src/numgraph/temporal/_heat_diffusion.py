from typing import Tuple, Callable, Optional, List
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
import numpy as np


def _heat_graph_diffusion(generator: Callable,
                          t_max: Optional[int] = None, 
                          heat_spike: Optional[float] = 1.,
                          init_temp: Optional[float] = None, 
                          num_nodes: Optional[int] = None,
                          step_size: float = 0.1,
                          return_coo: float = True,
                          rng: Optional[Generator] = None) -> Tuple[List[NDArray], List[NDArray]]:
    
    assert init_temp is None or (isinstance(init_temp, float) and init_temp > 0), f'init_temp can be None or float > 0, not {type(init_temp)} with value {init_temp}' 
    assert heat_spike is None or (isinstance(heat_spike, float) and heat_spike > 0), f'heat_spike can be None or float > 0, not {type(heat_spike)} with value {heat_spike}' 

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
    else:
        x = np.full((num_nodes,1), init_temp)

    # Improve heat of a random node
    i = rng.integer(num_nodes)
    x[i,0] = heat_spike if heat_spike is not None else rng.uniform(low=0.7, high=2.0, size=(num_nodes, 1))
    
    # Compute the Laplacian matrix
    adj_mat = np.zeros((num_nodes, num_nodes))
    adj_mat[edges[:, 0], edges[:, 1]] = 1 if weights is None else weights
    self_loops = np.diag(adj_mat)
    np.fill_diagonal(adj_mat, 0)
    degree = np.diag(np.sum(adj_mat, axis=1))
    L = np.eye(num_nodes) - degree^(-1/2) @ adj_mat @ degree^(-1/2) # Normalized laplacian

    xs = []
    for t in range(t_max):
        # Graph Heat Equation (Euler's method)
        x = x + step_size * (L @ x)
        xs.append(x)

    if return_coo:
        return [(edges, weights)] * t_max, xs
    else:
        adj_mat += self_loops 
        return [adj_mat] * t_max, xs


def heat_graph_diffusion_coo(generator: Callable,
                             t_max: Optional[int] = 10, 
                             heat_spike: Optional[float] = None,
                             init_temp: Optional[float] = None, 
                             step_size: float = 0.1,
                             num_nodes: Optional[int] = None,
                             rng: Optional[Generator] = None) -> Tuple[List[Tuple[NDArray, NDArray]], List[NDArray]]:
    """
    Returns heat diffusion over a graph. The model simulates the diffusion of heat on a given graph
    through the graph heat equation. Each node is characterized by the temperature. 
    The process is defined on discrite time, and last until t_max time is reached. 
    The simulation graph has fixed nodes and edges along the temporal axis.

    Parameters
    ----------
    generator : Callable
        A callable that takes as input a rng and generates the simulation graph
    t_max : int, optional
        The maximum number of timesteps in the simulation, by default 10
    heat_spike : float, optional
        The temperature of a spiking node. If None it computes a random temperature between 0.7 and 2., by default None
    init_temp : float, optional
        The initial temperature of the nodes. If None it computes a random temperature between 0. and 0.2, by default None
    ste_size : float, optional
        The step size used in the Euler's method discretization, by default 0.1
    num_nodes : int, optional
        The number of nodes in the simulation graph, by default None
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    Tuple
        The heat diffusion over the graph. The tuple contains 2 elements:
        - the list of graph's snapshots (T x 2): each snapshot is a tuple containing the 
           graph in COO representation (snapshot_num_edges x 2) and the weights (snapshot_num_edges x 1)
        - the list of nodes' states (T x (snapshot_num_nodes, ))
    """
    return _heat_graph_diffusion(generator = generator,  
                                 t_max = t_max, 
                                 heat_spike = heat_spike,
                                 init_temp = init_temp, 
                                 num_nodes = num_nodes,
                                 step_size = step_size,
                                 return_coo = True,
                                 rng = rng)


def heat_graph_diffusion_full(generator: Callable,
                              t_max: Optional[int] = None, 
                              heat_spike: Optional[float] = None,
                              init_temp: Optional[float] = None, 
                              step_size: float = 0.1,
                              num_nodes: Optional[int] = None,
                              rng: Optional[Generator] = None) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Returns heat diffusion over a graph. The model simulates the diffusion of heat on a given graph
    through the graph heat equation. Each node is characterized by the temperature. 
    The process is defined on discrite time, and last until t_max time is reached. 
    The simulation graph has fixed nodes and edges along the temporal axis.

    Parameters
    ----------
    generator : Callable
        A callable that takes as input a rng and generates the simulation graph
    t_max : int, optional
        The maximum number of timesteps in the simulation, by default 10
    heat_spike : float, optional
        The temperature of a spiking node. If None it computes a random temperature between 0.7 and 2., by default None
    init_temp : float, optional
        The initial temperature of the nodes. If None it computes a random temperature between 0. and 0.2, by default None
    ste_size : float, optional
        The step size used in the Euler's method discretization, by default 0.1
    num_nodes : int, optional
        The number of nodes in the simulation graph, by default None
    rng : Generator, optional
        Numpy random number generator, by default None

    Returns
    -------
    Tuple
        The heat diffusion over the graph. The tuple contains 2 elements:
        - the list of graph's snapshots in matrix representation (T x (num_nodes x num_nodes))
        - the list of nodes' states (T x (num_nodes, ))
    """
    return _heat_graph_diffusion(generator = generator,  
                                 t_max = t_max, 
                                 heat_spike = heat_spike,
                                 init_temp = init_temp, 
                                 num_nodes = num_nodes,
                                 step_size = step_size,
                                 return_coo = False,
                                 rng = rng)

