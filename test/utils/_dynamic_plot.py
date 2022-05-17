from matplotlib import animation
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import numpy as np
import tqdm

class DynamicHeatmap:

    def __init__(self, xs, shape, annot=False):
        self.fig, self.ax = plt.subplots()
        self.anim = None
        self.pbar = tqdm.tqdm(total=len(xs))
        self.xs = xs
        self.shape = shape
        self.annot = annot
        self.cmap = sns.color_palette("rocket", as_cmap=True)
        
        self.M, self.m = -np.inf, np.inf
        for x in xs:
            self.M = max(np.max(x), self.M)
            self.m = min(np.min(x), self.m)

    def animate(self):
        def init():
            sns.heatmap(np.zeros(self.shape), annot=self.annot, cmap=self.cmap, #linewidths=.5, 
                        yticklabels=False, xticklabels=False, cbar=False, ax=self.ax,
                        vmin=self.m, vmax=self.M)

        def animate(i):
            self.pbar.update(1)
            self.ax.texts = []
            x = self.xs[i].reshape(self.shape)
            sns.heatmap(x, annot=self.annot, cmap=self.cmap, #linewidths=.5, 
                        yticklabels=False, xticklabels=False, cbar=False, ax=self.ax,
                        vmin=self.m, vmax=self.M)

        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init, frames=len(self.xs), repeat=False)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=self.m, vmax=self.M))
        sm.set_array([])
        plt.colorbar(sm)



class DynamicHeatGraph:

    def __init__(self, edges, xs, layout):
        self.fig, self.ax = plt.subplots()
        self.anim = None
        self.pbar = tqdm.tqdm(total=len(xs))
        self.xs = xs
        self.edges = edges[0][0]
        self.G = nx.DiGraph()
        self.G.add_edges_from(self.edges)
        self.pos = layout(self.G)

        self.cmap = sns.color_palette("rocket", as_cmap=True) #magma
        self.M, self.m = -np.inf, np.inf
        for x in xs:
            self.M = max(np.max(x), self.M)
            self.m = min(np.min(x), self.m)

    def animate(self):
        def init():
            nx.draw(self.G, self.pos, node_color=self.xs[0], with_labels=True,
                    cmap=self.cmap, arrowstyle='-|>', vmin=self.m, vmax=self.M)
            
        def animate(i):
            self.pbar.update(1)
            nx.draw(self.G, self.pos, node_color=self.xs[i], with_labels=True,
                    cmap=self.cmap, arrowstyle='-|>', vmin=self.m, vmax=self.M)

        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init, frames=len(self.xs), repeat = False)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=self.m, vmax=self.M))
        sm.set_array([])
        plt.colorbar(sm)
