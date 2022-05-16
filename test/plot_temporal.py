from matplotlib import animation
from utils import DynamicHeatmap, DynamicGraph
from numgraph.distributions import *
from numgraph.temporal import *
import matplotlib.pyplot as plt
import networkx as nx


# Dissemination Process Simulation
print('Susceptible-Infected')
num_nodes = 15
generator = lambda _: clique_coo(num_nodes)
pos = nx.shell_layout(nx.from_edgelist(clique_coo(num_nodes)[0]))

snapshots, xs = susceptible_infected_coo(generator, prob=0.4, mask_size=0.8, t_max = 100)

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



# Heat Diffusion simulation
print('Heat diffusion')
h, w = 3, 3
generator = lambda _: simple_grid_coo(h,w, directed=False)
snapshots, xs = heat_graph_diffusion_coo(generator, t_max=100, num_nodes=h*w)

dh = DynamicHeatmap(xs=xs, shape=(h,w), annot=True)
dh.animate()
plt.show()
#f = "./DynamicHeatmap.mp4" 
#writervideo = animation.FFMpegWriter(fps=5) 
#dh.anim.save(f, writer=writervideo)


dh = DynamicGraph(edges=snapshots, xs=xs, layout=lambda G: nx.spring_layout(G, iterations=100, seed=9))
dh.animate()
plt.show()
#f = "./DynamicGraph.mp4" 
#writervideo = animation.FFMpegWriter(fps=5) 
#dh.anim.save(f, writer=writervideo)