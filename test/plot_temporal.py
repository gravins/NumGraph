from matplotlib import animation
from utils import DynamicHeatmap, DynamicHeatGraph, DynamicNodeSignal
from numgraph.distributions import *
from numgraph.temporal import *
import matplotlib.pyplot as plt
import networkx as nx
from numgraph.utils.spikes_generator import ColdHeatSpikeGenerator


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
plt.close()

# Heat Diffusion simulation (Euler's method)
print("Euler's method heat diffusion")
h, w = 3, 3
generator = lambda _: simple_grid_coo(h,w, directed=False)
t_max = 150
spikegen = ColdHeatSpikeGenerator(t_max=t_max, prob_cold_spike=0.5, num_spikes=10)
snapshots, xs = euler_graph_diffusion_coo(generator, spikegen, diffusion=None, t_max=t_max, num_nodes=h*w)

dh = DynamicHeatmap(xs=xs, shape=(h,w), annot=True)
dh.animate()
#plt.show()
f = "./EulerDynamicHeatmap.mp4" 
writervideo = animation.FFMpegWriter(fps=5) 
dh.anim.save(f, writer=writervideo)


dh = DynamicHeatGraph(edges=snapshots, xs=xs, layout=lambda G: nx.spring_layout(G, iterations=100, seed=9))
dh.animate()
#plt.show()
f = "./EulerDynamicGraph.mp4" 
writervideo = animation.FFMpegWriter(fps=5) 
dh.anim.save(f, writer=writervideo)


dh = DynamicNodeSignal(xs=xs)
dh.animate()
#plt.show()
f = "./EulerDynamicNodesignal.mp4" 
writervideo = animation.FFMpegWriter(fps=5) 
dh.anim.save(f, writer=writervideo)



# Heat Diffusion simulation (Closed form solution)
print("Closed form solution heat diffusion")
h, w = 3, 3
generator = lambda _: simple_grid_coo(h,w, directed=False)
t_max = 150
timestamps = [0.01 * i for i in range(100)]
snapshots, xs = heat_graph_diffusion_coo(generator, timestamps, num_nodes=h*w)

dh = DynamicHeatmap(xs=xs, shape=(h,w), annot=True)
dh.animate()
#plt.show()
f = "./DynamicHeatmap.mp4" 
writervideo = animation.FFMpegWriter(fps=5) 
dh.anim.save(f, writer=writervideo)


dh = DynamicHeatGraph(edges=snapshots, xs=xs, layout=lambda G: nx.spring_layout(G, iterations=100, seed=9))
dh.animate()
#plt.show()
f = "./DynamicGraph.mp4" 
writervideo = animation.FFMpegWriter(fps=5) 
dh.anim.save(f, writer=writervideo)


dh = DynamicNodeSignal(xs=xs)
dh.animate()
#plt.show()
f = "./DynamicNodesignal.mp4" 
writervideo = animation.FFMpegWriter(fps=5) 
dh.anim.save(f, writer=writervideo)