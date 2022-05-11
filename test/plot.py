import community as community_louvain
import networkx as nx
import matplotlib.pyplot as plt
from plot_utils import community_layout
import pydot
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