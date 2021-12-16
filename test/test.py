from numgraph.distributions import *
import networkx as nx
from plot import *
from numpy.random import default_rng

seed = 7

# Erdos-Renyi
print('Erdos-Renyi')
num_nodes = 10
prob = 0.4
G = erdos_renyi(num_nodes, prob)
G = nx.from_edgelist(G)
plot_er(G, num_nodes)

# SBM
print('SBM')
block_size = [15, 5, 3]
probs = [[0.5, 0.01, 0.01], [0.01, 0.5, 0.01], [0.01, 0.01, 0.5]]
generator = lambda b, p, rng: erdos_renyi(b, p)
G = stochastic_block_model(block_size, probs, generator, rng = default_rng(seed))
G = nx.from_edgelist(G)
plot_sbm(G, seed=seed)

# Barabasi Albert
print('Barabasi Albert')
num_nodes = 10
num_edges = 7
rng = default_rng(seed)
G = barabasi_albert(num_nodes, num_edges, rng)
G = nx.from_edgelist(G)
plot_ba(G, seed)

# Clique
print('Clique')
num_nodes = 10
G = clique(num_nodes)
G = nx.from_edgelist(G)
plot_clique(G)

# Star
print('Star')
num_nodes = 10
G = star(num_nodes)
G = nx.from_edgelist(G)
plot_star(G)

# Simple Grid
print('Simple Grid')
height, width = 3, 5
G = simple_grid(height, width)
G = nx.from_edgelist(G)
plot_grid(G)

# Full Grid
print('Full Grid')
height, width = 3, 5
G = full_grid(height, width)
G = nx.from_edgelist(G)
plot_grid(G)

# Random Tree
print('Random tree')
num_nodes = 10
G = random_tree(num_nodes, rng = default_rng(seed))
G = nx.from_edgelist(G)
plot_tree_on_terminal(G)