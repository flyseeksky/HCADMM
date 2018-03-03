#!/usr/local/bin/python3
import logging
from operator import itemgetter
import scipy.sparse as sps
import numpy as np
import scipy.linalg as LA
import networkx as nx
logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Simulator():
    def __init__(self, graph, hyper_edges=None, mode='', simulation_setting=None):
        assert isinstance(graph, nx.Graph), 'graph must be a networkx Graph object'
        self.graph = graph
        self.hyper_edge_list = hyper_edges
        self.mode = mode
        self.simulation_setting = simulation_setting

    def get_incidence(self, auto_discover=True):
        """
        Get incidence matrix for current graph with given mode.

        :param mode: specify how the algorithm would be run, must be one of 
                     hybrid, centralized, decentralized.
        :param auto_discover: only valid for hybrid mode, controlling 
                  whether or not to use auto discovery hyper edges algorithm
        :return: incidence matrix
        """
        assert self.mode != '', 'You need to set mode first'
        if self.mode == 'hybrid':
            return self._get_h_incidence(auto_discover)
        elif self.mode == 'centralized':
            return self._get_c_incidence()
        elif self.mode == 'decentralized':
            return self._get_d_incidence()
        else:
            raise Exception('Unsupported mode!')

    def _get_h_incidence(self, auto_discover=True):
        if auto_discover:
            # TODO auto threshold
            self.auto_discover_hyper_edge(2)
        return self.incidence_from_hyper_edge_list()

    def _get_c_incidence(self):
        return np.ones((self.graph.number_of_nodes(),1))

    def _get_d_incidence(self):
        return nx.incidence_matrix(self.graph)

    def incidence_from_hyper_edge_list(self):
        """
        Get incidence matrix from hyper-edge list.

        The shape of incidence matrix should be NxM, where N is number of 
        nodes, and M number of hyper-edges.
        C[i,j] = 1 if node i in hyper-edge j.
        :return: incidence matrix
        """
        assert self.hyper_edge_list, 'You need to set hyper_edge_list first'
        hyper_edge_list = sorted(self.hyper_edge_list)
        number_of_edges = len(hyper_edge_list)
        number_of_nodes = self.graph.number_of_nodes()

        incidence = sps.lil_matrix((number_of_nodes, number_of_edges))
        for edge_idx, edge in enumerate(hyper_edge_list):
            incidence[edge, edge_idx] = 1
        return sps.csr_matrix(incidence)

    def auto_discover_hyper_edge(self, threshold):
        """
        Automatically find patterns to build a hybrid model.
        """
        assert isinstance(self.graph, nx.Graph), 'Simulator.graph must'
        'be instance of networkx.Graph'

        node_degree_list = sorted(list(self.graph.degree), 
                                  key=itemgetter(1), reverse=True)
        # TODO fix number of local centers
        # consider according to descending degree order
        node_degree_list_to_consider = [nd for nd in node_degree_list 
                                        if nd[1] >= threshold]
        qualified_node_set = set([nd[0] for nd in node_degree_list_to_consider])
        all_edge_set = set(self.graph.edges)

        hyper_edge_list = []
        remaining_edge_set = all_edge_set.copy()
        for node, _ in node_degree_list_to_consider:
            # if current node not qualify, next
            if node not in qualified_node_set:
                continue

            all_neighbors = tuple(sorted(nx.all_neighbors(self.graph, node)))

            # add current hyper-edge into hyper-edge list
            hyper_edge = sorted([node] + list(all_neighbors))
            hyper_edge_list.append(tuple(hyper_edge))

            # mark all neighbors as not qualified
            qualified_node_set.difference_update(hyper_edge)

            # removing all edges from edge list
            # remember removing all the edges bewteen nodes in one hyper-edge
            edges = set([(node1, node2) for node1 in hyper_edge for node2 in 
                         hyper_edge if node1 < node2])
            remaining_edge_set.difference_update(edges)

        # combining hyper edges and all remaining simple edges
        hyper_edge_list += list(remaining_edge_set)
        self.hyper_edge_list = sorted(hyper_edge_list)
        return self.hyper_edge_list

    def run_least_squares(self):
        logging.debug('============ Mode: ' + self.mode + ' =================')

        # extract simulation settings
        setting = self.simulation_setting
        c, v, x0, max_iter = (setting['penalty'], setting['objective'], 
                              setting['initial'], setting['max_iter'])
        x_opt = v.mean(axis=0)

        C = self.get_incidence()
        if self.mode == 'hybrid':
            print(C.toarray())
        A, B = self.incidence_to_ab(C)
        # node_degree and edge_degree are 1D vectors, to be compatible with
        # numpy broadcasting rules
        # reshape them as (-1, 1), which is a 2D vector
        node_degree, edge_degree = ( np.squeeze(np.asarray(C.sum(axis=1))), 
                                    np.squeeze(np.asarray(C.sum(axis=0))) )

        # the reason that there is no need to expand C, A, B to block 
        # structures is because of this equality:
        # AXB = (A kron B^T) vec(X)
        # therefore, z = C^T x = C^T vec(X) = (C kron I)^T vec(X) = C^T X I = C^T X

        # initial value
        z0 = C.T.dot(x0) / edge_degree.reshape(-1, 1)
        alpha0 = np.zeros_like(x0)
        primal_gap, primal_residual, dual_residual = [], [], []

        logging.debug('Mode: %s, starting for-loop', self.mode)
        x, z, alpha = x0, z0, alpha0
        for i in range(max_iter):
            z_prev = z  # save for computing dual residual
            x = (v - alpha + c * C.dot(z)) / (1 + c*node_degree.reshape(-1, 1))
            z = C.T.dot(x) / edge_degree.reshape(-1, 1)
            alpha += c * (node_degree.reshape(-1, 1) * x - C.dot(z))

            primal_gap.append(LA.norm(x - x_opt) / LA.norm(x_opt))
            primal_residual.append(LA.norm(A.dot(x) - B.dot(z)))
            dual_residual.append(LA.norm(c * C.dot(z - z_prev)))

            # debug printing
            step = max_iter // 10
            if i % step == step - 1:
                logging.debug('Progress %.1f', 100 * (i+1)/ max_iter)

        logging.debug('Mode: %s, ending for loop', self.mode)
        return primal_gap, primal_residual, dual_residual

    @staticmethod
    def erdos_renyi(n_nodes, prob, seed=None):
        """
        randomly generate a connected graph using Erdos-Renyi model
        :param n_nodes: number of nodes
        :param prob: the probability of an edge
        :return: an Networkx object
        """

        G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
        return G

    @staticmethod
    def incidence_to_ab(incidence):
        """
        Generate matrix A,B from incidence matrix
        :param incidence:
        :return:
        """
        assert isinstance(incidence, np.ndarray) or sps.isspmatrix(incidence), 'Invalid incidence matrix'

        # Convert incidence matrix to sparse format
        if not sps.isspmatrix(incidence):
            sp_incidence = sps.csr_matrix(incidence)
        else:
            sp_incidence = incidence

        n, m = sp_incidence.shape
        t = sp_incidence.nnz

        # Elements of A, B need to be changed, so lil_matrix is more preferable
        A, B = sps.lil_matrix((t, n)), sps.lil_matrix((t, m))
        row_index, col_index = sp_incidence.nonzero()
        for idx, (row, col) in enumerate(zip(row_index, col_index)):
            A[idx, row] = 1
            B[idx, col] = 1
        return A.tocsr(), B.tocsr()
# ================================================================================================
# end of class definition
# ================================================================================================

# def check_hyper_edges(incidence, hyper_edges):
#     """
#     Check the consistency of incidence matrix and hyper edges.
#     To be consistent, conditions that must be satisfied:
#     1. incidence matrix is an instance of numpy array with dim = 2;
#     2. hyper_edges is a list of lists;
#     3. total number of nodes in hyper_edges equals that of incidence matrix
#     :param incidence: incidence matrix of underlying simple graph
#     :param hyper_edges: list of hyper edges, each of which specified by a list of nodes
#     :return: None
#     """
#     assert isinstance(incidence, np.ndarray) and incidence.ndim == 2
#     assert isinstance(hyper_edges, list)
#     # make all hyper_edges instance of list
#     hyper_edges = make_hyperedge_list(hyper_edges)
#     # verify total number of nodes in hyperedges are consistent
#     # to avoid duplicate counting, nodes should be identified uniquely
#     n_edges = incidence.shape[1]
#     total_edges = set()
#     for edge in hyper_edges:
#             total_edges.update(edge)
#     assert total_edges == set(range(n_edges))


# def make_hyperedge_list(hyper_edges):
#     for (n, edge) in enumerate(hyper_edges):
#         if not isinstance(edge, list):
#             hyper_edges[n] = [edge]
#     return hyper_edges
#
# def node_to_edge(incidence, node_list):
#     """
#     Convert hyper identifiers from node list to edge list
#     :param incidence: adjacency matrix for simple graph
#     :param node_list: nodes list of each hyper edge
#     :return:
#     """
#     edge_list = []
#     adj = np.array(incidence)
#     all_edges = np.arange(adj.shape[1])
#     for nodes in node_list:
#         sub_adj = adj[nodes, :]
#         edge_idx = sub_adj.sum(axis=0) > 1
#         edge_list.append(list(all_edges[edge_idx]))
#     return edge_list


# def hyper_incidence(incidence, hyper_edges):
#     """
#     This function convert an incidence matrix of a simple graph to an incidence of hypergraph, whose
#     edges specified by parameter hyper_edges
#     :param incidence: a matrix
#     :param hyper_edges: a list of hyper edges, each of which is a list of nodes
#     :return:
#     """
#     hyper_edges = make_hyperedge_list(hyper_edges)
#     check_hyper_edges(incidence, hyper_edges)
#     hyper_incidence_matrix = np.zeros((incidence.shape[0], len(hyper_edges)), dtype=np.int)
#     for n, edge in enumerate(hyper_edges):  # assuming edges are ascending ordered
#         if isinstance(edge, list) and len(edge) > 1: # if edge is a list, sub_incidence is a ndarray
#             sub_incidence = incidence[:, edge]
#             sub_hyper_incidence = sub_incidence.sum(axis=1) > 0
#         else:                                        # if edge is a single number, sub_incidence is a vector
#             sub_hyper_incidence = incidence[:, edge]
#         hyper_incidence_matrix[:, n] = sub_hyper_incidence.astype(np.int).ravel()
#     return hyper_incidence_matrix

# def get_AB(self):
#     assert self.hyper_edges is not None
#     M = len(self.hyper_edges)   # number of hyper-edges
#     N = self.n_nodes            # number of nodes
#     edge_degree = np.array([len(edge) for edge in self.hyper_edges])
#     full_list = []
#     for edge in self.hyper_edges:
#         full_list += edge
#     node_degree = np.array(list(dict(Counter(full_list)).values()))
#     assert node_degree.sum() == edge_degree.sum()
#
#     I_N, I_M = np.eye(N), np.eye(M)
#     A, B = [], []
#     for iN in range(N):
#         A += [I_N[iN]] * node_degree[iN]
#     A = np.array(A)
#     for iM in range(M):
#         B += [I_M[iM]] * edge_degree[iM]
#     B = np.array(B)
#     return A, B

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
# def index_to_position(index, size):
#     """
#     Convert a list of index to positional vector.
#
#     Index begins from zero. Each element of index will correspond to one '1' in positional vector.
#
#     >>>> index_to_position([2, 3], 5)
#     array([0, 0, 1, 1, 0])
#     >>>> index_to_position([1], 3)
#     array([0, 1, 0])
#     :param index: list of index
#     :param size: size of positional vector
#     :return: positional vector
#     """
#     assert max(index) < size, 'Index out of bound'
#     pos_vec = np.zeros((size))
#     for i in index:
#         pos_vec[i] = 1
#     return pos_vec
# def from_hyper_edges(self):
#     hyper_edges = self.hyper_edges
#     edge_degree = np.array([len(edge) for edge in hyper_edges])
#     total_degree = edge_degree.sum()
#     number_of_edges = len(hyper_edges)
#     number_of_nodes = self.graph.number_of_nodes()
#     I_A, I_B = np.eye(number_of_nodes), np.eye(number_of_edges)
#
#     A, B = [], []
#     for idx, edge in enumerate(hyper_edges):
#         B += [list(I_B[idx])] * len(edge)
#         for node in edge:
#             A.append(list(I_A[node]))
#     #
#     # A = sorted(A, reverse=True)
#     # B = [b for a,b in sorted(zip(A, B), key=itemgetter(0), reverse=True)]
#     A, B = np.array(A), np.array(B)
#     return A, B