Usage
=====

Here we provide some examples regarding how to use ``numgraph``.


NumGraph in 2 steps
-------------------

.. code:: python


   >>> from numgraph import star_coo, star_full
   >>> coo_matrix, coo_weights = star_coo(num_nodes=5, weighted=True)
   >>> print(coo_matrix)
   array([[0, 1],
          [0, 2],
          [0, 3],
          [0, 4],
          [1, 0],
          [2, 0],
          [3, 0],
          [4, 0]]

   >>> print(coo_weights)
   array([[0.89292422],
          [0.3743427 ],
          [0.32810002],
          [0.97663266],
          [0.74940571],
          [0.89292422],
          [0.3743427 ],
          [0.32810002],
          [0.97663266],
          [0.74940571]])

   >>> adj_matrix = star_full(num_nodes=5, weighted=True)
   >>> print(adj_matrix)
   array([[0.        , 0.72912008, 0.33964166, 0.30968042, 0.08774328],
          [0.72912008, 0.        , 0.        , 0.        , 0.        ],
          [0.33964166, 0.        , 0.        , 0.        , 0.        ],
          [0.30968042, 0.        , 0.        , 0.        , 0.        ],
          [0.08774328, 0.        , 0.        , 0.        , 0.        ]])

Other examples can be found in `plot_static.py <https://github.com/gravins/NumGraph/blob/main/test/plot_static.py>`_  and `plot_temporal.py <https://github.com/gravins/NumGraph/blob/main/test/plot_temporal.py>`_.


Stochastic Block Model
----------------------

To generate a Stochastic Block Model graph, we need to define from which distribution the communities belong.  
Moreover, we need to identify the probability matrix: 
-   the element ``i,j`` represents the edge probability between blocks i and j
-   the element ``i,i`` define the edge probability inside block i. 


.. code:: python


   >>> from numgraph import stochastic_block_model_coo
   >>> block_sizes = [15, 5, 3]
   >>> probs = [[0.5, 0.01, 0.01], 
   >>>          [0.01, 0.5, 0.01], 
   >>>          [0.01, 0.01, 0.5]]
   >>> generator = lambda b, p, rng: erdos_renyi_coo(b, p)
   >>> coo_matrix, coo_weights = stochastic_block_model_coo(block_sizes = block_sizes, 
   >>>                                                      probs = probs, 
   >>>                                                      generator = generator)
   

.. note::
   The communities are generated with consecutive node ids. Let consider the previous example where ``block_sizes = [15, 5, 3]``. Here the first community has node ids in ``[0,15)``, the second in ``[15,20)``, and the third in ``[20,23)``.


Heat Diffusion simulation
-------------------------

Similarly to the SBM generation, even the temporal distribution require the definition of a generator to compute the employed graph. In the case of the heat diffusion simulation it also important to define a ``SpikeGenerator``, which specifies how heat spikes are generated over time.


.. code:: python


   >>> from numgraph import simple_grid_coo
   >>> from numgraph.utils.spikes_generator import ColdHeatSpikeGenerator
   >>> from numgraph.temporal import heat_graph_diffusion_coo
   
   >>> h, w = 3, 3
   >>> generator = lambda _: simple_grid_coo(h, w, directed=False)
   >>> t_max = 150
    
   >>> spikegen = ColdHeatSpikeGenerator(t_max=t_max, prob_cold_spike=0.5, num_spikes=10)
   >>> snapshots, xs = heat_graph_diffusion_coo(generator, spikegen, t_max=t_max, num_nodes=h*w)
   >>> print(snapshots[0]) # the topology of the graph at time 0
   array([[0., 1., 0., 1., 0., 0., 0., 0., 0.],
          [1., 0., 1., 0., 1., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0., 1., 0., 0., 0.],
          [1., 0., 0., 0., 1., 0., 1., 0., 0.],
          [0., 1., 0., 1., 0., 1., 0., 1., 0.],
          [0., 0., 1., 0., 1., 0., 0., 0., 1.],
          [0., 0., 0., 1., 0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 1., 0., 1., 0., 1.],
          [0., 0., 0., 0., 0., 1., 0., 1., 0.]])
   
   >>> print(xs[0]) # the temperature of each node at time 0
   array([[ 0.10196489],
          [-0.17995079],
          [ 0.04456628],
          [ 0.05386166],
          [ 0.03761498],
          [ 0.040233  ],
          [ 0.09440064],
          [ 0.17265226],
          [ 0.15886457]])
