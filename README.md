[pypi-image]: https://github.com/gravins/NumGraph/blob/main/img/NumGraph_logo.svg
[pypi-url]: https://pypi.org/project/numgraph/

<p align="center">
  <img width="30%" src="https://github.com/gravins/NumGraph/blob/main/img/NumGraph_logo.svg" />
</p>


# NumGraph
Numpy library for synthetic graph generation.

## Implemented distributions
### Static Graphs
- Star graph
- Clique
- Two-dimensional rectangular grid lattice graph
- Random Tree
- Erdos Renyi
- Barabasi Albert
- Stochastic Block Model

### Temporal Graphs
- Susceptible-Infected Dissemination Process Simulation
- Heat diffusion over a graph

## Installation

``` python3 -m pip install numgraph ```

## Usage
```python

>>> from numgraph.distributions import star_coo, star_full
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

```

