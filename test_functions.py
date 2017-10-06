from Admm_simulator import *
import networkx as nx
import numpy as np


class TestAutoDiscovery(object):
    def test_star_graph(self):
        g = Simulator(nx.star_graph(10 - 1))
        hyper_edges = g.auto_discover_hyper_edge(np.random.randint(2, 9))
        assert hyper_edges == [tuple(range(10))]

        hyper_edges = g.auto_discover_hyper_edge(10)
        assert set(hyper_edges) == {(0, i) for i in range(1, 10)}

    def test_line_graph(self):
        g = nx.path_graph(10)
        hyper_edges = Simulator(g).auto_discover_hyper_edge(2)
        assert set(hyper_edges) == {(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8), (8, 9)}

    def test_hybrid_graph(self):
        g = nx.Graph()
        g.add_edges_from([(0,1),(1,2),(1,3),(3,4)])
        sim = Simulator(g)
        hyper_edge_list = sim.auto_discover_hyper_edge(2)
        assert set(hyper_edge_list) == {(0,1,2,3), (3,4)}

        hyper_edge_list = sim.auto_discover_hyper_edge(4)
        assert set(hyper_edge_list) == {(0,1),(1,2),(1,3),(3,4)}


class TestIncidenceFromHyperEdge(object):
    def test_line_graph(self):
        hyper_edge_list = [(0,1,2,3,4,5,6,7,8,9)]


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

    def test_incidence_to_AB(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])
        C = nx.incidence_matrix(g)
        A, B = incidence_to_ab(C.todense())
        AA = np.array([[1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1]])
        BB = np.array([[1, 0, 0, 0],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 1]])
        assert np.all(A == AA)
        assert np.all(B == BB)

            # def test_incidence_to_AB_another(self):
            #     g = nx.Graph().add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])


# class TestAB(object):
#     def test_AB_linegraph(self):
#         gg = Simulator(4, [[0, 1, 2], [2, 3]])
#         # gg.line_graph()
#         A, B = gg.get_AB()
#         AA = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#         BB = np.array([[1, 0]] * 3 + [[0, 1]] * 2)
#         assert np.all(A == AA)
#         assert np.all(B == BB)
#
#
#     def test_AB_stargraph(self):
#         gg = Simulator(4, [[0, 1, 2, 3]])
#         # gg.star_graph()
#         A, B = gg.incidence_from_hyper_edges()
#         assert np.all(A == np.eye(4))
#         assert np.all(B == np.ones(4, ))
#
#
#     def test_AB_hybridgraph(self):
#         gg = Simulator(6, [[0, 1, 2, 3], [3, 4], [4, 5]])
#         A, B = gg.get_AB()
#         AA = np.array([[1, 0, 0, 0, 0, 0],
#                        [0, 1, 0, 0, 0, 0],
#                        [0, 0, 1, 0, 0, 0],
#                        [0, 0, 0, 1, 0, 0],
#                        [0, 0, 0, 1, 0, 0],
#                        [0, 0, 0, 0, 1, 0],
#                        [0, 0, 0, 0, 1, 0],
#                        [0, 0, 0, 0, 0, 1]])
#         BB = np.array([[1, 0, 0]] * 4 + [[0, 1, 0]] * 2 + [[0, 0, 1]] * 2)
#         assert np.all(A == AA)
#         assert np.all(B == BB)
#
#     def test_AB_unordered(self):
#         gg = Simulator(5, [[0, 1, 2], [1,4], [2,3]])
#         A, B = gg.incidence_from_hyper_edges()
#         BB = np.array([[1, 0, 0],
#                        [1, 0, 0],
#                        [1, 0, 0],
#                        [0, 1, 0],
#                        [0, 1, 0],
#                        [0, 0, 1],
#                        [0, 0, 1]])
#         AA = np.array([[1, 0, 0, 0, 0],
#                        [0, 1, 0, 0, 0],
#                        [0, 0, 1, 0, 0],
#                        [0, 1, 0, 0, 0],
#                        [0, 0, 0, 0, 1],
#                        [0, 0, 1, 0, 0],
#                        [0, 0, 0, 1, 0]])
#         assert np.all(A == AA)
#         assert np.all(B == BB)
#
#     def test_centralized_AB(self):
#         gs = Simulator(n_nodes=5)
#         c_hyper_edges = gs._get_c_incidence()
#         assert c_hyper_edges == [[0,1,2,3,4]]
#
#     def test_d_AB(self):
#         g = nx.Graph()
#         g.add_nodes_from(range(5))
#         g.add_edges_from([(0,1),(1,2),(1,3),(3,4)])
#         A, B = Simulator._get_d_incidence(g)
#         AA = np.array([[1, 0, 0, 0, 0],
#                        [0, 1, 0, 0, 0],
#                        [0, 1, 0, 0, 0],
#                        [0, 1, 0, 0, 0],
#                        [0, 0, 1, 0, 0],
#                        [0, 0, 0, 1, 0],
#                        [0, 0, 0, 1, 0],
#                        [0, 0, 0, 0, 1]])
#         BB = np.array([[1, 0, 0, 0],
#                        [1, 0, 0, 0],
#                        [0, 1, 0, 0],
#                        [0, 0, 1, 0],
#                        [0, 1, 0, 0],
#                        [0, 0, 1, 0],
#                        [0, 0, 0, 1],
#                        [0, 0, 0, 1]])
#
#         assert np.all(A == AA), 'A not equal'
#         assert np.all(B == BB), 'B not equal'
#

