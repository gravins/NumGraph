from numgraph.temporal_distributions import *
from numgraph.distributions import *
import matplotlib.pyplot as plt
import networkx as nx

# Dissemination Process Simulation
print('Susceptible-Infected')
num_nodes = 15
generator = lambda _: clique(num_nodes)
pos = nx.shell_layout(nx.from_edgelist(clique(num_nodes)))
print(pos)

snapshots, xs = susceptible_infected(generator, prob=0.4, mask_size=0.8, t_max = 100)

for edge_list, x in zip(snapshots, xs):
    nodes = [(i, {'color': 'red' if v else 'green'}) for i, v in enumerate(x)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    nodes = G.nodes()
    sorted(nodes)
    c = nx.get_node_attributes(G,'color')
    colors = [c[n] for n in nodes]
    nx.draw(G, pos=pos, node_color=colors, arrowstyle='-|>')

    plt.pause(0.5)
    plt.clf() 
