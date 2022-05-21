from numgraph.distributions import *
from ._plot import plot_ba, plot_clique, plot_er, plot_grid, plot_sbm, plot_star, plot_tree, plot_tree_on_terminal
from ._dynamic_plot import DynamicHeatGraph, DynamicHeatmap, DynamicNodeSignal

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
 
