from email.policy import default
import numpy as np
from numgraph.distributions import *
import unittest
from utils import get_all_matrices, get_directional_matrices, get_weighted_matrices


N = 10
block_size = [10, 5, 3]
probs = [[0.5, 0.01, 0.01], [0.01, 0.5, 0.01], [0.01, 0.01, 0.5]]
generator = lambda b, p, rng: erdos_renyi_coo(b, p)
w = h = 3
p = 0.35
seed = 7
rng1 = np.random.default_rng(seed)
rng2 = np.random.default_rng(seed)

class TestStaticGraphDim(unittest.TestCase):

    def test_full_dim(self):
        matrices = get_all_matrices(N, p, block_size, probs, h, w, generator, rng1, coo=False)

        for num_nodes, matrix, _ in matrices:
            row, col = matrix.shape
            self.assertTrue(row == col and row == num_nodes)


    def test_full_weights(self):
        matrices = get_weighted_matrices(N, p, block_size, probs, h, w, generator, rng1, coo=False)

        for _, matrix, _ in matrices:
            self.assertTrue(np.all(matrix <= 1))


    def test_full_directed(self):
        matrices = get_directional_matrices(True, N, p, block_size, probs, h, w, generator, rng1, coo=False)

        for i, (_, matrix, name) in enumerate(matrices):
            j = 0
            while j < 10 and np.all(matrix == matrix.T): # check if return always an undirected graph or it is only an unluky sampling
                rng = np.random.default_rng(2**j)
                matrix = get_directional_matrices(True, N, p, block_size, probs, h, w, generator, rng, coo=False)[i]
                j += 1
            #print(name)
            self.assertTrue(not np.all(matrix == matrix.T))


    def test_full_undirected(self):
        matrices = get_directional_matrices(False, N, p, block_size, probs, h, w, generator, rng1, coo=False)

        for _, matrix, name in matrices:
            #print(f'{name}:the matrix is', matrix, matrix == matrix.T, np.all(matrix == matrix.T), sep='\n')
            #print('\n\n')
            self.assertTrue(np.all(matrix == matrix.T))


    def test_full_deterministic_sampling(self):
        matrices1 = get_all_matrices(N, p, block_size, probs, h, w, generator, rng1, coo=False)
        matrices2 = get_all_matrices(N, p, block_size, probs, h, w, generator, rng2, coo=False)

        for (_, matrix1, name), (_, matrix2, _)  in zip(matrices1, matrices2):
            print(name)
            self.assertTrue(np.all(matrix1 == matrix2))


    def test_coo_dim(self):
        matrices = get_all_matrices(N, p, block_size, probs, h, w, generator, rng1, coo=True)

        for _, matrix, _  in matrices:
            self.assertTrue(isinstance(matrix, tuple))
            coo_matrix, coo_weights  = matrix
            row1, col1 = coo_matrix.shape
            self.assertTrue(col1 == 2)
            if coo_weights is not None:
                row2, col2 = coo_weights.shape
                self.assertTrue(row1 == row2 and col2 == 1)


    def test_coo_deterministic_sampling(self):
        matrices1 = get_all_matrices(N, p, block_size, probs, h, w, generator, rng1, coo=True)
        matrices2 = get_all_matrices(N, p, block_size, probs, h, w, generator, rng2, coo=True)

        for (_, matrix1, name1), (_, matrix2, name2)  in zip(matrices1, matrices2):
            #print("\n\n\n", name1, name2)
            #print(matrix1[0].shape, matrix2[0].shape)
            #print(np.all(matrix1[0] == matrix2[0]))
            self.assertTrue(isinstance(matrix1, tuple) and isinstance(matrix2, tuple))
            self.assertTrue(np.all(matrix1[0] == matrix2[0])) # check edges
            self.assertTrue(np.all(matrix1[1] == matrix2[1])) # check weights



if __name__ == '__main__':
    unittest.main(verbosity=2)