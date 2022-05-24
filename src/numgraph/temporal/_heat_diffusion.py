from typing import Tuple, Callable, Optional, List
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
import numpy as np
from numgraph.utils.spikes_generator import SpikeGenerator

def _heat_graph_diffusion(generator: Callable,
                          spike_generator: SpikeGenerator,
                          t_max: int = 10, 
                          init_temp: Optional[float] = None, 
                          num_nodes: Optional[int] = None,
                          step_size: float = 0.1,
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
    else:
        x = np.full((num_nodes,1), init_temp)
    
    # Compute the Laplacian matrix
    adj_mat = np.zeros((num_nodes, num_nodes))
    adj_mat[edges[:, 0], edges[:, 1]] = 1 if weights is None else weights
    self_loops = np.diag(adj_mat)
    np.fill_diagonal(adj_mat, 0)
    degree = np.diag(np.sum(adj_mat, axis=1))
    new_degree = np.linalg.inv(np.sqrt(degree))
    L = np.eye(num_nodes) - new_degree @ adj_mat @ new_degree # Normalized laplacian

    xs = []
    for t in range(t_max):
        x = spike_generator.compute_spike(t, x)
        
        # Graph Heat Equation (Euler's method)
        xs.append(x)
        x = x + step_size * (-L @ x)

    if return_coo:
        return [(edges, weights)] * t_max, xs
    else:
        adj_mat += self_loops 
        return [adj_mat] * t_max, xs


def heat_graph_diffusion_coo(generator: Callable,
                             spike_generator: SpikeGenerator,
                             t_max: int = 10, 
                             init_temp: Optional[float] = None, 
                             step_size: float = 0.1,
                             num_nodes: Optional[int] = None,
                             rng: Optional[Generator] = None) -> Tuple[List[Tuple[NDArray, NDArray]], List[NDArray]]:
    """
    Returns heat diffusion over a graph. The model simulates the diffusion of heat on a given graph
    through the graph heat equation. Each node is characterized by the temperature. 
    The process is defined on discrite time, and last until :obj:`t_max` time is reached. 
    The simulation graph has fixed nodes and edges along the temporal axis.

    Parameters
    ----------
    generator : Callable
        A callable that takes as input a rng and generates the simulation graph
    spike_generator : SpikeGenerator
        The spike generator, which implement the method :obj:`compute_spike(t, x)`
    t_max : int, optional
        The maximum number of timesteps in the simulation, by default :obj:`10`
    init_temp : float, optional
        The initial temperature of the nodes. If :obj:`None` it computes a random temperature between :obj:`0.` and :obj:`0.2`, by default :obj:`None`
    step_size : float, optional
        The step size used in the Euler's method discretization, by default :obj:`0.1`
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
                                 spike_generator = spike_generator,
                                 t_max = t_max, 
                                 init_temp = init_temp, 
                                 num_nodes = num_nodes,
                                 step_size = step_size,
                                 return_coo = True,
                                 rng = rng)


def heat_graph_diffusion_full(generator: Callable,
                              spike_generator: SpikeGenerator,
                              t_max: int = 10,
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
    spike_generator : SpikeGenerator
        The spike generator, which implement the method :obj:`compute_spike(t,x)`
    t_max : int, optional
        The maximum number of timesteps in the simulation, by default :obj:`10`
    init_temp : float, optional
        The initial temperature of the nodes. If :obj:`None` it computes a random temperature between :obj:`0.` and :obj:`0.2`, by default :obj:`None`
    step_size : float, optional
        The step size used in the Euler's method discretization, by default :obj:`0.1`
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
                                 spike_generator = spike_generator,
                                 t_max = t_max, 
                                 init_temp = init_temp, 
                                 num_nodes = num_nodes,
                                 step_size = step_size,
                                 return_coo = False,
                                 rng = rng)
