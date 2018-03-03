import Hypergraph as hg
import numpy as np
import networkx as nx
import pytest


@pytest.fixture()
def example1():
    C = np.array([[1, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1]])
    H = hg.Hypergraph(C)
    return H


def test_incidence(example1):
    Ch = example1.incidence_matrix()
    Ce = np.array([[1,1,1,1,0,0],[0,0,0,1,1,1]]).T
    assert np.all(Ce == Ch)


def test_neighbor(example1):
    Nh = example1.neighbor(3)
    Ne = np.arange(6)
    assert np.all(Nh == Ne)

    Nh = example1.neighbor(5)
    Ne = np.array([3,4,5])
    assert np.all(Nh == Ne)


def test_line_5():
    G = nx.path_graph(5)
    C = np.asarray(nx.incidence_matrix(G).todense())
    H = hg.Hypergraph(C)
    Ch = H.incidence_matrix()
    Ce = np.array([[1,1,1,0,0],[0,0,1,1,1]]).T
    assert np.all(Ce == Ch)


def test_line_6():
    G = nx.path_graph(6)
    C = np.asarray(nx.incidence_matrix(G).todense())
    H = hg.Hypergraph(C)
    Ch = H.incidence_matrix()
    Ce = np.array([[1, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 1, 1]]).T
    assert np.all(Ce == Ch)


def test_star():
    n = np.random.randint(2, 100)
    G = nx.star_graph(n)
    C = np.asarray(nx.incidence_matrix(G).todense())
    H = hg.Hypergraph(C)
    Ch = H.incidence_matrix()
    Ce = np.ones((n+1, 1))
    assert np.all(Ce == Ch)