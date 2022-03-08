import numpy as np
from src.numgraph.distributions import erdos_renyi_coo, erdos_renyi_full


def test_full_dim():
    N = 4
    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1)

    row, col = matrix.shape
    assert row == col and row == N

def test_full_weights():
    N = 16
    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.3,
                              directed=True,
                              weighted=True)

    assert np.all(matrix <= 1)


    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.3,
                              directed=False,
                              weighted=True)

    assert np.all(matrix <= 1)



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
    N = 16
    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1,
                              directed=False)

    assert np.all(matrix == matrix.T)

    matrix = erdos_renyi_full(num_nodes=N,
                              prob=0.1,
                              directed=False,
                              weighted=True)

    assert np.all(matrix == matrix.T)


def test_full_deterministic_sampling():
    N = 16
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


def test_coo_dim():
    N = 4

    coo_matrix, _  = erdos_renyi_coo(num_nodes=N, prob=0.1)
    row, col = coo_matrix.shape
    assert col == 2

    coo_matrix, coo_weights  = erdos_renyi_coo(num_nodes=N, prob=0.1, weighted=True)
    row1, col1 = coo_matrix.shape
    row2, col2 = coo_weights.shape

    assert row1 == row2 and col2 == 1


def test_coo_deterministic_sampling():
    N = 16
    seed = 7
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    coo_matrix1, _ = erdos_renyi_coo(num_nodes=N,
                                     prob=0.1,
                                     directed=False,
                                     rng=rng1)

    coo_matrix2, _ = erdos_renyi_coo(num_nodes=N,
                                     prob=0.1,
                                     directed=False,
                                     rng=rng2)

    assert np.all(coo_matrix1 == coo_matrix2)
