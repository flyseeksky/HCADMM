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
    B = hyper_incidence(A, [[0, 1, 2], [3]])
    C = np.array([[1, 0], [1, 0], [1, 0], [1, 1], [0, 1]])
    C2 = np.array([[1], [1], [1], [1], [1]])
    B1 = hyper_incidence(A, [0, 1, 2, 3])
    B2 = hyper_incidence(A, [[0, 1, 2, 3]])
    assert np.all(B == C)
    assert np.all(B1 == A)
    assert np.all(B2 == C2)


def test_nodetoedgelist():
    A = np.array([[1, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])
    node_list = [[0, 1, 2, 3], [3, 4]]
    edge_list = node_to_edge(A, node_list)
    target = [[0, 1, 2], [3]]
    assert edge_list == target


def test_convertadj():
    A = np.array([[1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 1],
                  [0, 0, 1]])
    edge_list = [[0, 1], [2]]
    hyper_A = hyper_incidence(A, edge_list)
    target = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])
    assert np.all(hyper_A == target)