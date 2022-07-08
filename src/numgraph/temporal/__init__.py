from ._susceptible_infected import susceptible_infected_coo, susceptible_infected_full
from ._heat_diffusion import heat_graph_diffusion_coo, heat_graph_diffusion_full
from ._euler_diffusion import euler_graph_diffusion_coo, euler_graph_diffusion_full

__all__ = [
    'susceptible_infected_coo',
    'susceptible_infected_full',
    'heat_graph_diffusion_coo',
    'heat_graph_diffusion_full',
    'euler_graph_diffusion_coo',
    'euler_graph_diffusion_full'

]