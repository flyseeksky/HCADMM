from GraphGenerator import *
import networkx as nx
import numpy as np


def test_connected():
    G = random_connected_graph(100, .1)
    assert nx.is_connected(G)

def test_hyper_adj():
    A = np.array([[1, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])
    B = simple_adj_to_hyper_adj(A, [[0,1,2], [3]])
    C = np.array([[1, 0], [1, 0], [1, 0], [1, 1], [0, 1]])
    C2 = np.array([[1], [1], [1], [1], [1]])
    B1 = simple_adj_to_hyper_adj(A, [0, 1, 2, 3])
    B2 = simple_adj_to_hyper_adj(A, [[0, 1, 2, 3]])
    assert np.all(B == C)
    assert np.all(B1 == A)
    assert np.all(B2 == C2)