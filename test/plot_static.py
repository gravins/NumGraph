from numgraph.distributions import *
import networkx as nx
from utils import *
from numpy.random import default_rng

seed = 7

# Erdos-Renyi
print('Erdos-Renyi')
num_nodes = 10
prob = 0.4
e, _ = erdos_renyi_coo(num_nodes, prob)
G = nx.DiGraph()
G.add_edges_from(e)
plot_er(G, num_nodes)

# SBM
print('SBM')
block_size = [15, 5, 3]
probs = [[0.5, 0.01, 0.01], [0.01, 0.5, 0.01], [0.01, 0.01, 0.5]]
generator = lambda b, p, rng: erdos_renyi_coo(b, p)
e, _ = stochastic_block_model_coo(block_size, probs, generator, rng = default_rng(seed))
G = nx.from_edgelist(e)
plot_sbm(G, seed=seed)

# Barabasi Albert
print('Barabasi Albert')
num_nodes = 10
num_edges = 7
rng = default_rng(seed)
e, _ = barabasi_albert_coo(num_nodes, num_edges, rng)
G = nx.DiGraph()
G.add_edges_from(e)
plot_ba(G, seed)

# Clique
print('Clique')
num_nodes = 10
e, _ = clique_coo(num_nodes)
G = nx.DiGraph()
G.add_edges_from(e)
plot_clique(G)

# Star
print('Star')
num_nodes = 10
e, _ = star_coo(num_nodes)
G = nx.DiGraph()
G.add_edges_from(e)
plot_star(G)

# Simple Grid
print('Simple Grid')
height, width = 3, 5
e, _ = simple_grid_coo(height, width)
G = nx.DiGraph()
G.add_edges_from(e)
plot_grid(G)

# Full Grid
print('Full Grid')
height, width = 3, 5
e, _ = grid_coo(height, width)
G = nx.DiGraph()
G.add_edges_from(e)
plot_grid(G)

# Random Tree
print('Random tree')
num_nodes = 10
e, _ = random_tree_coo(num_nodes, rng = default_rng(seed))
G = nx.DiGraph()
G.add_edges_from(e)
plot_tree_on_terminal(G)
plot_tree(G)