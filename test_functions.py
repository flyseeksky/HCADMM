from GraphSimulator import *
import networkx as nx
import numpy as np


# def test_connected():
#     gg = ConsensusSimulator(n_nodes=100)
#     G = gg.erdos_renyi(.1)
#     assert nx.is_connected(G)
class TestUtilities(object):
    def test_hyper_incidence(self):
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


    def test_nodetoedgelist(self):
        A = np.array([[1, 0, 0, 0],
                      [1, 1, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]])
        node_list = [[0, 1, 2, 3], [3, 4]]
        edge_list = node_to_edge(A, node_list)
        target = [[0, 1, 2], [3]]
        assert edge_list == target


    def test_convertadj(self):
        A = np.array([[1, 0, 0],
                      [1, 1, 0],
                      [0, 1, 1],
                      [0, 0, 1]])
        edge_list = [[0, 1], [2]]
        hyper_A = hyper_incidence(A, edge_list)
        target = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])

        assert np.all(hyper_A == target)

class TestAB(object):
    def test_AB_linegraph(self):
        gg = Simulator(4, [[0, 1, 2], [2, 3]])
        # gg.line_graph()
        A, B = gg.get_AB()
        AA = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        BB = np.array([[1, 0]] * 3 + [[0, 1]] * 2)
        assert np.all(A == AA)
        assert np.all(B == BB)


    def test_AB_stargraph(self):
        gg = Simulator(4, [[0, 1, 2, 3]])
        # gg.star_graph()
        A, B = gg.from_hyper_edges()
        assert np.all(A == np.eye(4))
        assert np.all(B == np.ones(4, ))


    def test_AB_hybridgraph(self):
        gg = Simulator(6, [[0, 1, 2, 3], [3, 4], [4, 5]])
        A, B = gg.get_AB()
        AA = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
        BB = np.array([[1, 0, 0]] * 4 + [[0, 1, 0]] * 2 + [[0, 0, 1]] * 2)
        assert np.all(A == AA)
        assert np.all(B == BB)

    def test_AB_unordered(self):
        gg = Simulator(5, [[0, 1, 2], [1,4], [2,3]])
        A, B = gg.from_hyper_edges()
        BB = np.array([[1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1]])
        AA = np.array([[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0]])
        assert np.all(A == AA)
        assert np.all(B == BB)


class TestAutoDiscovery(object):
    # test star graph
    def test_auto_star_graph(self):
        g = nx.star_graph(10 - 1)
        hyper_edges = Simulator.auto_discover_hyper_edge(g)
        assert hyper_edges == [list(range(10))]

    # test line graph
    def test_line_graph(self):
        g = nx.path_graph(10)
        hyper_edges = Simulator.auto_discover_hyper_edge(g)
        assert hyper_edges == [[0,1,2],[2,3,4],[4,5,6],[6,7,8],[8,9]]
