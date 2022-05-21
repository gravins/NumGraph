from typing import Tuple, Optional
from numpy.random import Generator, default_rng
import numpy as np


class SpikeGenerator:
    """
    This class is responsible of the generation of the heat spikes
    """
    def __init__(self,
                 t_max: int = 10, 
                 heat_spike: Tuple[float, float] = (0.7, 2.),
                 num_spikes: int = 1,
                 rng: Optional[Generator] = None) -> None:
        """
        The generator of the spikes in the heat diffusion over a graph. To each spike is associated an
        increase of temperature of the node. This class identify at each step the node with injected 
        temperature and the new temperature.
        Parameters
        ----------
        t_max : int, optional
            The maximum number of timesteps in the simulation, by default 10
        heat_spike : Tuple[float, float], optional
            A tuple containing the min and max temperature of a spike, by default (0.7, 2.)
        num_spikes : int, optional
            The number of heat spikes during the process, by default 1
        rng : Generator, optional
            Numpy random number generator, by default None
        """
        assert ((isinstance(heat_spike, tuple) and 
                isinstance(heat_spike[0], float) and
                isinstance(heat_spike[0], float)), 
                f'heat_spike can be Tuple[float, float], not {heat_spike}')

        assert num_spikes > 0 and num_spikes < t_max, 'num_spike must be in the range (0, t_max)'

        self.t_max = t_max
        self.heat_spike = heat_spike
        self.num_spikes = num_spikes
        if rng is None:
            self.rng = default_rng()
        else:
            self.rng = rng

        self.spike_timesteps = set([0])
        if num_spikes > 1:
            tmp = np.arange(1, t_max)
            self.rng.shuffle(tmp)
            self.spike_timesteps |= set(tmp[:self.num_spikes])

    def compute_spike(self, t, x):
        raise NotImplementedError()


class HeatSpikeGenerator:
    def __init__(self,
                 t_max: int = 10, 
                 heat_spike: Tuple[float, float] = (0.7, 2.),
                 num_spikes: int = 1,
                 rng: Optional[Generator] = None) -> None:
        """
        The generator of the spikes in the heat diffusion over a graph. To each spike is associated an
        increase of temperature of the node. This class identify at each step the node with injected 
        temperature and the new temperature.
        Parameters
        ----------
        t_max : int, optional
            The maximum number of timesteps in the simulation, by default 10
        heat_spike : Tuple[float, float], optional
            A tuple containing the min and max temperature of a spike, by default (0.7, 2.)
        num_spikes : int, optional
            The number of heat spikes during the process, by default 1
        rng : Generator, optional
            Numpy random number generator, by default None
        """
        super().__init__(t_max, heat_spike, num_spikes, rng)

    def compute_spike(self, t, x):
        if t in self.spike_timesteps:
            # Improve heat of a random node
            i = self.rng.integers(x.shape[0])
            x[i,0] = self.rng.uniform(low=self.heat_spike[0], 
                                      high=self.heat_spike[1],
                                      size=(1, 1))
        return x

class ColdHeatSpikeGenerator(SpikeGenerator):

    def __init__(self,
                 t_max: int = 10, 
                 heat_spike: Tuple[float, float] = (0.7, 2.),
                 cold_spike: Tuple[float, float] = (-1., 0.),
                 prob_cold_spike: float = 0.2,
                 num_spikes: int = 2,
                 rng: Optional[Generator] = None) -> None:
        """
        The generator of the spikes in the heat diffusion over a graph. To each spike is associated an
        increase (or decrease) of temperature of the node. This class identify at each step the node with injected 
        temperature and the new temperature.
        Parameters
        ----------
        t_max : int, optional
            The maximum number of timesteps in the simulation, by default 10
        heat_spike : Tuple[float, float], optional
            A tuple containing the min and max temperature of a hot spike, by default (0.7, 2.)
        cold_spike : Tuple[float, float], optional
            A tuple containing the min and max temperature of a cold spike, by default (-1., 0.)
        prob_cold_spike : float, optional
            The probability of a cold spike to happen, by default 0.2
        num_spikes : int, optional
            The number of heat spikes during the process, by default 1
        rng : Generator, optional
            Numpy random number generator, by default None
        """
        super().__init__(t_max, heat_spike, num_spikes, rng)
        assert prob_cold_spike > 0 and prob_cold_spike < 1, 'prob_cold_spike if a probability in the range (0, 1)'  
        
        self.cold_spike = cold_spike
        self.prob_cold_spike = prob_cold_spike

    def compute_spike(self, t, x):
        if t in self.spike_timesteps:
            # Improve heat of a random node
            i = self.rng.integers(x.shape[0])
            if self.rng.uniform(low=0, high=1) < self.prob_cold_spike:
                x[i,0] = self.rng.uniform(low=self.cold_spike[0], 
                                          high=self.cold_spike[1],
                                          size=(1, 1))
            else:
                x[i,0] = self.rng.uniform(low=self.heat_spike[0], 
                                          high=self.heat_spike[1],
                                          size=(1, 1))
        return x