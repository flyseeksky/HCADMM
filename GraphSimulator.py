import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from collections import Counter
import numpy.linalg as LA


class Simulator():
    def __init__(self, n_nodes, hyper_edges=None, v=None, max_iter=100, penalty=10):
        self.n_nodes = n_nodes
        self.hyper_edges = hyper_edges
        self.v = v
        self.max_iter = max_iter
        self.c = penalty

    def get_AB(self):
        assert self.hyper_edges is not None
        M = len(self.hyper_edges)   # number of hyper-edges
        N = self.n_nodes            # number of nodes
        edge_degree = np.array([len(edge) for edge in self.hyper_edges])
        full_list = []
        for edge in self.hyper_edges:
            full_list += edge
        node_degree = np.array(list(dict(Counter(full_list)).values()))
        assert node_degree.sum() == edge_degree.sum()

        I_N, I_M = np.eye(N), np.eye(M)
        A, B = [], []
        for iN in range(N):
            A += [I_N[iN]] * node_degree[iN]
        A = np.array(A)
        for iM in range(M):
            B += [I_M[iM]] * edge_degree[iM]
        B = np.array(B)
        return A, B

    def run_least_squares(self, n_iter, x0):
        c, v = self.c, self.v
        A, B = self.get_AB()
        D_M = B.T.dot(B)
        D_N = A.T.dot(A)
        C = A.T.dot(B)
        x_star = v.mean()

        z0 = LA.pinv(D_M).dot(C.T).dot(x0)
        alpha0 = np.zeros_like(x0)
        primal_gap = []
        primal_residual, dual_residual = [], []
        x, z, alpha = x0, z0, alpha0
        for i in range(n_iter):
            z_prev = z
            x = LA.pinv(np.eye(self.n_nodes) + c * D_N).dot(v - alpha + c * C.dot(z))
            z = LA.inv(D_M).dot(C.T).dot(x)
            alpha += c * (D_N.dot(x) - C.dot(z))

            primal_gap.append(LA.norm(x - x_star) / LA.norm(x_star))
            primal_residual.append(LA.norm(A.dot(x) - B.dot(z)) + np.finfo(float).eps)
            dual_residual.append(LA.norm(c * C.dot(z - z_prev)) + np.finfo(float).eps)
        return primal_gap, primal_residual, dual_residual


def check_hyper_edges(incidence, hyper_edges):
    """
    Check the consistency of incidence matrix and hyper edges.
    To be consistent, conditions that must be satisfied:
    1. incidence matrix is an instance of numpy array with dim = 2;
    2. hyper_edges is a list of lists;
    3. total number of nodes in hyper_edges equals that of incidence matrix
    :param incidence: incidence matrix of underlying simple graph
    :param hyper_edges: list of hyper edges, each of which specified by a list of nodes
    :return: None
    """
    assert isinstance(incidence, np.ndarray) and incidence.ndim == 2
    assert isinstance(hyper_edges, list)
    # make all hyper_edges instance of list
    hyper_edges = make_hyperedge_list(hyper_edges)
    # verify total number of nodes in hyperedges are consistent
    # to avoid duplicate counting, nodes should be identified uniquely
    n_edges = incidence.shape[1]
    total_edges = set()
    for edge in hyper_edges:
            total_edges.update(edge)
    assert total_edges == set(range(n_edges))


def make_hyperedge_list(hyper_edges):
    for (n, edge) in enumerate(hyper_edges):
        if not isinstance(edge, list):
            hyper_edges[n] = [edge]
    return hyper_edges

def node_to_edge(incidence, node_list):
    """
    Convert hyper identifiers from node list to edge list
    :param incidence: adjacency matrix for simple graph
    :param node_list: nodes list of each hyper edge
    :return:
    """
    edge_list = []
    adj = np.array(incidence)
    all_edges = np.arange(adj.shape[1])
    for nodes in node_list:
        sub_adj = adj[nodes, :]
        edge_idx = sub_adj.sum(axis=0) > 1
        edge_list.append(list(all_edges[edge_idx]))
    return edge_list


def hyper_incidence(incidence, hyper_edges):
    """
    This function convert an incidence matrix of a simple graph to an incidence of hypergraph, whose
    edges specified by parameter hyper_edges
    :param incidence: a matrix
    :param hyper_edges: a list of hyper edges, each of which is a list of nodes
    :return:
    """
    hyper_edges = make_hyperedge_list(hyper_edges)
    check_hyper_edges(incidence, hyper_edges)
    hyper_incidence_matrix = np.zeros((incidence.shape[0], len(hyper_edges)), dtype=np.int)
    for n, edge in enumerate(hyper_edges):  # assuming edges are ascending ordered
        if isinstance(edge, list) and len(edge) > 1: # if edge is a list, sub_incidence is a ndarray
            sub_incidence = incidence[:, edge]
            sub_hyper_incidence = sub_incidence.sum(axis=1) > 0
        else:                                        # if edge is a single number, sub_incidence is a vector
            sub_hyper_incidence = incidence[:, edge]
        hyper_incidence_matrix[:, n] = sub_hyper_incidence.astype(np.int).ravel()
    return hyper_incidence_matrix

# deleted during refactoring
    # def erdos_renyi(self, prob):
    #     """
    #     randomly generate a connected graph using Erdos-Renyi model
    #     :param n_nodes: number of nodes
    #     :param prob: the probability of an edge
    #     :return: an Networkx object
    #     """
    #
    #     G = nx.erdos_renyi_graph(self.n_nodes, prob)
    #     while not nx.is_connected(G):
    #         G = nx.erdos_renyi_graph(self.n_nodes, prob)
    #     self.info = 'Erdos Renyi'
    #     self.graph = G
    #     return G

    # def line_graph(self):
    #     """
    #     Generate a line graph
    #     :return: networkx graph object
    #     """
    #     self.graph = nx.path_graph(self.n_nodes)
    #     self.info = 'line graph'
    #     return self.graph

    # def star_graph(self):
    #     """
    #     Generate a star graph
    #     :return: networkx graph object
    #     """
    #     self.graph = nx.star_graph(self.n_nodes - 1)
    #     self.info = 'star graph'
    #     return self.graph
    #
    # def cluster_graph(self, n_clusters=2):
    #     """
    #     Generate a graph with clusters
    #     :param n_clusters: number of clusters
    #     :return: networkx graph object
    #     """
    #     #
