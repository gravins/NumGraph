import numpy as np
from numgraph.distributions import *
import community as community_louvain
import networkx as nx
import matplotlib.pyplot as plt
from community_layout_utils import community_layout
from networkx.drawing.nx_pydot import graphviz_layout

def plot_er(G, n):
  """
  Plots the Erdos-Renyi graph

  Parameters
  ----------
  G: nx.Graph
    The ER graph
  n: int
    Number of nodes
  """
  # Put the nodes in a circular shape
  pos = nx.circular_layout(G)
  nx.draw(G, pos)
  plt.show()


def plot_sbm(G, seed):
  """
  Plots the Stochastic Block Model graph

  Parameters
  ----------
  G: nx.Graph
    The SBM graph
  seed: int
    random seed
  """
  partition = community_louvain.best_partition(G, random_state=seed)
  pos = community_layout(G, partition)
  nx.draw(G, pos, node_color=list(partition.values()))
  plt.show()


def plot_ba(G, seed):
  """
  Plots the Barabasi Albert Model graph
  Parameters
  ----------
  G: nx.Graph
    The BA graph
  seed: int
    random seed
  """
  pos = nx.circular_layout(G)
  nx.draw(G, pos)
  #partition = community_louvain.best_partition(G, random_state=seed)
  #pos = community_layout(G, partition)
  #nx.draw(G, pos, node_color=list(partition.values()))
  plt.show()


def plot_tree_on_terminal(G):
  """
  Plots the random tree graph

  Parameters
  ----------
  G: nx.Graph
    The random tree graph
  """
  print(nx.forest_str(G))


def plot_tree(G):
  """
  Plots the random tree graph

  Parameters
  ----------
  G: nx.Graph
    The random tree graph
  """
  pos = graphviz_layout(G, prog="twopi")
  nx.draw(G, pos)
  plt.show()


def plot_star(G):
  """
  Plots the star graph

  Parameters
  ----------
  G: nx.Graph
    The star graph
  """
  pos = nx.planar_layout(G)
  nx.draw(G, pos)
  plt.show()


def plot_clique(G):
  """
  Plots the clique graph

  Parameters
  ----------
  G: nx.Graph
    The clique graph
  """
  pos = nx.shell_layout(G)
  nx.draw(G, pos)
  plt.show()


def plot_grid(G):
  """
  Plots the simple grid graph

  Parameters
  ----------
  G: nx.Graph
    The simple grid graph
  """
  pos = nx.spring_layout(G, iterations=100, seed=9)
  nx.draw(G, pos)
  plt.show()


def get_directional_matrices(directed, N, p, block_size, probs, h, w, generator, rng, coo=False):
    if coo:
        m = [
            [N, erdos_renyi_coo(num_nodes=N, prob=p, directed=directed, weighted=True, rng=rng), f"erdos_renyi_coo(num_nodes={N}, prob={p}, directed={directed}, weighted=True, rng=rng)"],
            [N, erdos_renyi_coo(num_nodes=N, prob=p, directed=directed, weighted=False, rng=rng), f"erdos_renyi_coo(num_nodes={N}, prob={p}, directed={directed}, weighted=False, rng=rng)"],
            [N, star_coo(num_nodes=N, directed=directed, weighted=False, rng=rng), f"star_coo(num_nodes={N}, directed={directed}, weighted=False, rng=rng)"],
            [N, star_coo(num_nodes=N, directed=directed, weighted=True, rng=rng), f"star_coo(num_nodes={N}, directed={directed}, weighted=True, rng=rng)"],
            [sum(block_size), stochastic_block_model_coo(block_size, probs, generator, directed=directed, weighted=False, rng=rng), f"stochastic_block_model_coo(block_size={block_size}, probs={probs}, generator, directed={directed}, weighted=False, rng=rng)"],
            [sum(block_size), stochastic_block_model_coo(block_size, probs, generator, directed=directed, weighted=True, rng=rng), f"stochastic_block_model_coo(block_size={block_size}, probs={probs}, generator, directed={directed}, weighted=True, rng=rng)"],
            [N, random_tree_coo(num_nodes=N, directed=directed, weighted=False, rng=rng), f"random_tree_coo(num_nodes={N}, directed={directed}, weighted=False, rng=rng)"],
            [N, random_tree_coo(num_nodes=N, directed=directed, weighted=True, rng=rng), f"random_tree_coo(num_nodes={N}, directed={directed}, weighted=True, rng=rng)"]
        ]
        if not directed:
            m += [
                [N, clique_coo(num_nodes=N, weighted=False, rng=rng), f"clique_coo(num_nodes={N}, weighted=False, rng=rng)"],
                [N, clique_coo(num_nodes=N, weighted=True, rng=rng), f"clique_coo(num_nodes={N}, weighted=True, rng=rng)"],
                [w*h, grid_coo(height=h, width=w, weighted=False, rng=rng), f"grid_coo(height={h}, width={w}, weighted=False, rng=rng)"],
                [w*h, grid_coo(height=h, width=w, weighted=True, rng=rng), f"grid_coo(height={h}, width={w}, weighted=True, rng=rng)"],
                [w*h, simple_grid_coo(height=h, width=w, weighted=False, rng=rng), f"simple_grid_coo(height={h}, width={w}, weighted=False, rng=rng)"],
                [w*h, simple_grid_coo(height=h, width=w, weighted=True, rng=rng), f"simple_grid_coo(height={h}, width={w}, weighted=True, rng=rng)"],
                [N, barabasi_albert_coo(num_nodes=N, num_edges=int(N/2), weighted=False, rng=rng), f"barabasi_albert_coo(num_nodes={N}, num_edges={int(N/2)}, weighted=False, rng=rng)"], 
                [N, barabasi_albert_coo(num_nodes=N, num_edges=int(N/2), weighted=True, rng=rng), f"barabasi_albert_coo(num_nodes={N}, num_edges={int(N/2)}, weighted=True, rng=rng)"] 
            ]
    else:
        m = [
            [N, erdos_renyi_full(num_nodes=N, prob=p, directed=directed, weighted=True, rng=rng), f"erdos_renyi_full(num_nodes={N}, prob={p}, directed={directed}, weighted=True, rng=rng)"],
            [N, erdos_renyi_full(num_nodes=N, prob=p, directed=directed, weighted=False, rng=rng), f"erdos_renyi_full(num_nodes={N}, prob={p}, directed={directed}, weighted=False, rng=rng)"],
            [N, star_full(num_nodes=N, directed=directed, weighted=False, rng=rng), f"star_full(num_nodes={N}, directed={directed}, weighted=False, rng=rng)"],
            [N, star_full(num_nodes=N, directed=directed, weighted=True, rng=rng), f"star_full(num_nodes={N}, directed={directed}, weighted=True, rng=rng)"],
            [sum(block_size), stochastic_block_model_full(block_size, probs, generator, directed=directed, weighted=False, rng=rng), f"stochastic_block_model_full(block_size={block_size}, probs={probs}, generator, directed={directed}, weighted=False, rng=rng)"],
            [sum(block_size), stochastic_block_model_full(block_size, probs, generator, directed=directed, weighted=True, rng=rng), f"stochastic_block_model_full(block_size={block_size}, probs={probs}, generator, directed={directed}, weighted=True, rng=rng)"],
            [N, random_tree_full(num_nodes=N, directed=directed, weighted=False, rng=rng), f"random_tree_full(num_nodes={N}, directed={directed}, weighted=False, rng=rng)"],
            [N, random_tree_full(num_nodes=N, directed=directed, weighted=True, rng=rng), f"random_tree_full(num_nodes={N}, directed={directed}, weighted=True, rng=rng)"]
        ]

        if not directed:
            m += [
                [N, clique_full(num_nodes=N, weighted=False, rng=rng), f"clique_full(num_nodes={N}, weighted=False, rng=rng)"],
                [N, clique_full(num_nodes=N, weighted=True, rng=rng), f"clique_full(num_nodes={N}, weighted=True, rng=rng)"],
                [w*h, grid_full(height=h, width=w, weighted=False, rng=rng), f"grid_full(height={h}, width={w}, weighted=False, rng=rng)"],
                [w*h, grid_full(height=h, width=w, weighted=True, rng=rng), f"grid_full(height={h}, width={w}, weighted=True, rng=rng)"],
                [w*h, simple_grid_full(height=h, width=w, weighted=False, rng=rng), f"simple_grid_full(height={h}, width={w}, weighted=False, rng=rng)"],
                [w*h, simple_grid_full(height=h, width=w, weighted=True, rng=rng), f"simple_grid_full(height={h}, width={w}, weighted=True, rng=rng)"],
                [N, barabasi_albert_full(num_nodes=N, num_edges=int(N/2), weighted=False, rng=rng), f"barabasi_albert_full(num_nodes={N}, num_edges={int(N/2)}, weighted=False, rng=rng)"],
                [N, barabasi_albert_full(num_nodes=N, num_edges=int(N/2), weighted=True, rng=rng), f"barabasi_albert_full(num_nodes={N}, num_edges={int(N/2)}, weighted=True, rng=rng)"] 
            ]
    return m


def get_weighted_matrices(N, p, block_size, probs, h, w, generator, rng, coo=False):
    if coo:
        return [
            [N, erdos_renyi_coo(num_nodes=N, prob=p, directed=True, weighted=True, rng=rng), f"erdos_renyi_coo(num_nodes={N}, prob={p}, directed=True, weighted=True, rng=rng)"],
            [N, erdos_renyi_coo(num_nodes=N, prob=p, directed=False, weighted=True, rng=rng), f"erdos_renyi_coo(num_nodes={N}, prob={p}, directed=False, weighted=True, rng=rng)"],
            [N, clique_coo(num_nodes=N, weighted=True, rng=rng), f"clique_coo(num_nodes={N}, weighted=True, rng=rng)"],
            [w*h, grid_coo(height=h, width=w, weighted=True, rng=rng), f"grid_coo(height={h}, width={w}, weighted=True, rng=rng)"],
            [w*h, simple_grid_coo(height=h, width=w, weighted=True, rng=rng), f"simple_grid_coo(height={h}, width={w}, weighted=True, rng=rng)"],
            [N, star_coo(num_nodes=N, directed=True, weighted=True, rng=rng), f"star_coo(num_nodes={N}, directed=True, weighted=True, rng=rng)"],
            [N, star_coo(num_nodes=N, directed=False, weighted=True, rng=rng), f"star_coo(num_nodes={N}, directed=False, weighted=True, rng=rng)"],
            [sum(block_size), stochastic_block_model_coo(block_size, probs, generator, directed=True, weighted=True, rng=rng), f"stochastic_block_model_coo(block_size={block_size}, probs={probs}, generator, directed=True, weighted=True, rng=rng)"],
            [sum(block_size), stochastic_block_model_coo(block_size, probs, generator, directed=False, weighted=True, rng=rng), f"stochastic_block_model_coo(block_size={block_size}, probs={probs}, generator, directed=False, weighted=True, rng=rng)"],
            [N, barabasi_albert_coo(num_nodes=N, num_edges=int(N/2), weighted=True, rng=rng), f"barabasi_albert_coo(num_nodes={N}, num_edges={int(N/2)}, weighted=True, rng=rng)"],
            [N, random_tree_coo(num_nodes=N, directed=True, weighted=True, rng=rng), f"random_tree_coo(num_nodes={N}, directed=True, weighted=True, rng=rng)"],
            [N, random_tree_coo(num_nodes=N, directed=False, weighted=True, rng=rng), f"random_tree_coo(num_nodes={N}, directed=False, weighted=True, rng=rng)"] 
        ]
    else:
        return [
            [N, erdos_renyi_full(num_nodes=N, prob=p, directed=True, weighted=True, rng=rng), f"erdos_renyi_full(num_nodes={N}, prob={p}, directed=True, weighted=True, rng=rng)"],
            [N, erdos_renyi_full(num_nodes=N, prob=p, directed=False, weighted=True, rng=rng), f"erdos_renyi_full(num_nodes={N}, prob={p}, directed=False, weighted=True, rng=rng)"],
            [N, clique_full(num_nodes=N, weighted=True, rng=rng), f"clique_full(num_nodes={N}, weighted=True, rng=rng)"],
            [w*h, grid_full(height=h, width=w, weighted=True, rng=rng), f"grid_full(height={h}, width={w}, weighted=True, rng=rng)"],
            [w*h, simple_grid_full(height=h, width=w, weighted=True, rng=rng), f"simple_grid_full(height={h}, width={w}, weighted=True, rng=rng)"],
            [N, star_full(num_nodes=N, directed=True, weighted=True, rng=rng), f"star_full(num_nodes={N}, directed=True, weighted=True, rng=rng)"],
            [N, star_full(num_nodes=N, directed=False, weighted=True, rng=rng), f"star_full(num_nodes={N}, directed=False, weighted=True, rng=rng)"],
            [sum(block_size), stochastic_block_model_full(block_size, probs, generator, directed=True, weighted=True, rng=rng), f"stochastic_block_model_full(block_size={block_size}, probs={probs}, generator, directed=True, weighted=True, rng=rng)"],
            [sum(block_size), stochastic_block_model_full(block_size, probs, generator, directed=False, weighted=True, rng=rng), f"stochastic_block_model_full(block_size={block_size}, probs={probs}, generator, directed=False, weighted=True, rng=rng)"],
            [N, barabasi_albert_full(num_nodes=N, num_edges=int(N/2), weighted=True, rng=rng), f"barabasi_albert_full(num_nodes={N}, num_edges={int(N/2)}, weighted=True, rng=rng)"],
            [N, random_tree_full(num_nodes=N, directed=True, weighted=True, rng=rng), f"random_tree_full(num_nodes={N}, directed=True, weighted=True, rng=rng)"],
            [N, random_tree_full(num_nodes=N, directed=False, weighted=True, rng=rng), f"random_tree_full(num_nodes={N}, directed=False, weighted=True, rng=rng)"]
        ]

def get_all_matrices(N, p, block_size, probs, h, w, generator, rng, coo=False):

    return (get_directional_matrices(False, N, p, block_size, probs, h, w, generator, rng, coo=coo) + 
            get_directional_matrices(True, N, p, block_size, probs, h, w, generator, rng, coo=coo) +
            get_weighted_matrices(N, p, block_size, probs, h, w, generator, rng, coo=coo))
 
