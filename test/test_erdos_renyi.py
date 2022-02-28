import numpy as np
from src.numgraph.distributions import erdos_renyi_coo, erdos_renyi_full


def test_full_dim():
    N = 4
    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1)

    row, col = matrix.shape
    assert row == col and row == N


def test_full_weights():
    N = 4
    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1,
                              weighted=True)


def test_full_directed():
    N = 16
    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1,
                              directed=True)

    assert not np.all(matrix == matrix.T)

    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1,
                              directed=True,
                              weighted=True)

    assert not np.all(matrix == matrix.T)


def test_full_undirected():
    N = 4
    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1,
                              directed=False)

    assert np.all(matrix == matrix.T)

    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1,
                              directed=False,
                              weighted=True)

    assert np.all(matrix == matrix.T)


def test_full_rng():
    N = 4
    seed = 7
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    matrix1 = erdos_renyi_full(num_nodes=N,
                               prob=0.1,
                               directed=False,
                               rng=rng1)

    matrix2 = erdos_renyi_full(num_nodes=N,
                               prob=0.1,
                               directed=False,
                               rng=rng2)

    assert np.all(matrix1 == matrix2)
