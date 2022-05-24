from typing import Tuple, Callable, Optional, Union, List
from numpy.random import Generator, default_rng
from scipy.sparse import coo_matrix
from numpy.typing import NDArray
import numpy as np


def _susceptible_infected(generator: Callable, 
                          prob: float = 0.5, 
                          mask_size: float = 0.5, 
                          t_max: Optional[int] = None, 
                          infected_nodes: Union[int, float] = 0.1, 
                          num_nodes: Optional[int] = None,
                          rng: Optional[Generator] = None) -> Tuple[List[NDArray], List[NDArray]]:

    assert prob >= 0 and prob < 1, 'prob should be a probability in the range [0, 1)' 
    assert mask_size >= 0 and mask_size < 1, 'mask_size should be a probability in the range [0, 1)' 

    if rng is None:
        rng = default_rng()

    # Generate the graph
    edges = generator(rng) 
    assert isinstance(edges, tuple) and edges[0].shape[1] == 2, 'The generator must return a graph in COO representation.'
    edges = edges[0]

    if num_nodes is None:
        num_nodes = edges.max() + 1

    assert (infected_nodes > 0 and infected_nodes <= 1) or infected_nodes < num_nodes, \
           'infected_nodes should be a probability in the range (0, 1] or an integer < num_nodes'
    
    # Define the starting infected nodes
    x = np.zeros(num_nodes)
    infected_nodes = round(num_nodes * infected_nodes) if isinstance(infected_nodes, float) \
                     else infected_nodes
    x[:infected_nodes] = 1
    rng.shuffle(x)
    infected = x > 0

    # Propagete the infection
    max_idx = round((1 - mask_size) * edges.shape[0])
    t_max = np.inf if t_max is None else t_max
    snapshots, xs =  [], []
    t = 0
    while t < t_max and not all(x == 1):
        mask = np.arange(edges.shape[0])
        rng.shuffle(mask)
        mask = mask[:max_idx]

        A_t = edges[mask].T
        vals, row, col = np.ones(A_t.shape[1]), A_t[0], A_t[1] 
        A_t = coo_matrix((vals, (row, col)), shape=(num_nodes, num_nodes))

        x = ((rng.uniform() * A_t.dot(x)) > prob).astype(int)
        x[infected] = 1
        infected = x > 0
        
        # Store info at time t
        snapshots.append(edges[mask])
        xs.append(x)

        t += 1

    return snapshots, xs

def susceptible_infected_coo(generator: Callable, 
                             prob: float = 0.5, 
                             mask_size: float = 0.5, 
                             t_max: Optional[int] = None, 
                             infected_nodes: Union[int, float] = 0.1, 
                             num_nodes: Optional[int] = None,
                             rng: Optional[Generator] = None) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Returns Dissemination Process Simulation (DPS). The model simulates a susceptible-infected scenario,
    e.g., epidemic spreading. Each node can be either infected (1) or susceptible (0). A node can
    change its state to infected depending on the number of infected neighbors and a fixed 
    probability. When a node change its state, it stays infected indefinitely. 
    The disseminaiton process is defined on discrite time, and last until each node is infected or
    :obj:`t_max` is reached. The simulation graph has fixed nodes and dynamic edges along the temporal axis.

    Parameters
    ----------
    generator : Callable
        A callable that takes as input a rng and generates the simulation graph
    prob : float, optional
        The probability of infection, by default :obj:`0.5`
    mask_size : float, optional
        The amount of edges to discard at each timestep, by default :obj:`0.5`
    t_max : int, optional
        The maximum number of timesteps in the simulation, by default :obj:`None`
    infected_nodes : Union[int, float], optional
        The amount of starting infected nodes, by default :obj:`0.1`
    num_nodes : int, optional
        The number of nodes in the simulation graph, by default :obj:`None`
    rng : Generator, optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    List[NDArray]
        The list of graph's snapshots in COO representation :obj:`(T x (snapshot_num_edges x 2))`
    List[NDArray]
        The list of nodes' states :obj:`(T x (num_nodes, ))`

    Note
    ----
        The weights computed by the :obj:`generator` are not used by the function
    """
    return  _susceptible_infected(generator = generator, 
                                  prob = prob,
                                  mask_size = mask_size,
                                  t_max = t_max, 
                                  infected_nodes = infected_nodes, 
                                  num_nodes = num_nodes,
                                  rng = rng)
    


def susceptible_infected_full(generator: Callable, 
                             prob: float = 0.5, 
                             mask_size: float = 0.5, 
                             t_max: Optional[int] = None, 
                             infected_nodes: Union[int, float] = 0.1, 
                             num_nodes: Optional[int] = None,
                             rng: Optional[Generator] = None) -> Tuple[List[NDArray], List[NDArray]]:
    """
    Returns Dissemination Process Simulation (DPS). The model simulates a susceptible-infected scenario,
    e.g., epidemic spreading. Each node can be either infected :obj:`(1)` or susceptible :obj:`(0)`. A node can
    change its state to infected depending on the number of infected neighbors and a fixed 
    probability. When a node change its state, it stays infected indefinitely. 
    The disseminaiton process is defined on discrite time, and last until each node is infected or
    t_max is reached. The simulation graph has fixed nodes and dynamic edges along the temporal axis.

    Parameters
    ----------
    generator : Callable
        A callable that takes as input a rng and generates the simulation graph
    prob : float, optional
        The probability of infection, by default :obj:`0.5`
    mask_size : float, optional
        The amount of edges to discard at each timestep, by default :obj:`0.5`
    t_max : int, optional
        The maximum number of timesteps in the simulation, by default :obj:`None`
    infected_nodes : Union[int, float], optional
        The amount of starting infected nodes, by default :obj:`0.1`
    num_nodes : int, optional
        The number of nodes in the simulation graph, by default :obj:`None`
    rng : Generator, optional
        Numpy random number generator, by default :obj:`None`

    Returns
    -------
    List[NDArray]
        The list of graph's snapshots in matrix represnetation :obj:`(T x (num_nodes x num_nodes))`
    List[NDArray]
        The list of nodes' states :obj:`(T x (num_nodes, ))`

    Note
    ----
        The weights computed by the :obj:`generator` are not used by the function

    """
    edges_snapshots_coo, nodes_snapshots =  _susceptible_infected(generator = generator, 
                                                                  prob = prob,
                                                                  mask_size = mask_size,
                                                                  t_max = t_max, 
                                                                  infected_nodes = infected_nodes, 
                                                                  num_nodes = num_nodes,
                                                                  rng = rng)
    edges_snapshots = []
    for coo_matrix in edges_snapshots_coo:
        adj_matrix = np.zeros((num_nodes, num_nodes))
        adj_matrix[coo_matrix[:, 0], coo_matrix[:, 1]] = 1
        edges_snapshots.append(adj_matrix)

    return edges_snapshots, nodes_snapshots
