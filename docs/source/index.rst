:github_url: https://github.com/gravins/NumGraph

NumGraph
========

**Num(py)Graph** is a library for synthetic graph generation. The main principle of NumGraph is to be a lightweight library (i.e., ``numpy`` is the only dependency) that generates graphs from a broad range of distributions. Indeed, It implements several graph distributions in both the static and temporal domain. 


Implemented distributions
-------------------------

Static Graphs
~~~~~~~~~~~~~

-  Star graph
-  Clique
-  Two-dimensional rectangular grid lattice graph
-  Random Tree
-  Erdos Renyi
-  Barabasi Albert
-  Stochastic Block Model

Temporal Graphs
~~~~~~~~~~~~~~~

-  Susceptible-Infected Dissemination Process Simulation
-  Heat diffusion over a graph (closed form solution)
-  Generic Euler's method approximation of a diffusion process over a graph



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   numgraph.distributions
   numgraph.temporal
   numgraph.utils
