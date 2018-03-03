#!/usr/local/bin/python3
import logging
from operator import itemgetter
import scipy.sparse as sps
import numpy as np
import scipy.linalg as LA
import networkx as nx
from Hypergraph import *
logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Simulator():
    def __init__(self, graph, mode='', simulation_setting=None):
        assert isinstance(graph, nx.Graph), 'graph must be a networkx Graph object'
        self.graph = graph
        self.mode = mode
        self.simulation_setting = simulation_setting

    def run_least_squares(self):
        logging.debug('============ Mode: ' + self.mode + ' =================')

        # extract simulation settings
        setting = self.simulation_setting
        rho, v, x0, max_iter = (setting['penalty'], setting['objective'],
                                setting['initial'], setting['max_iter'])
        x_opt = v.mean(axis=0)

        if self.mode == 'centralized':
            C = np.ones((self.graph.number_of_nodes(), 1))
        elif self.mode == 'decentralized':
            C = nx.incidence_matrix(self.graph)
        elif self.mode == 'hybrid':
            C = np.asarray(nx.incidence_matrix(self.graph).todense())
            H = Hypergraph(C)
            C = H.incidence_matrix()
            print(C)
        else:
            raise ValueError("mode must be set")
        A, B = self.incidence_to_ab(C)

        # reshape them as (-1, 1), which is a 2D vector
        node_degree, edge_degree = (np.squeeze(np.asarray(C.sum(axis=1))),
                                    np.squeeze(np.asarray(C.sum(axis=0))))

        # initial value
        z0 = C.T.dot(x0) / edge_degree.reshape(-1, 1)
        alpha0 = np.zeros_like(x0)
        primal_gap, primal_residual, dual_residual = [], [], []

        logging.debug('Mode: %s, starting for-loop', self.mode)
        x, z, alpha = x0, z0, alpha0
        for i in range(max_iter):
            z_prev = z  # save for computing dual residual
            x = (v - alpha + rho * C.dot(z)) / (1 + rho*node_degree.reshape(-1, 1))
            z = C.T.dot(x) / edge_degree.reshape(-1, 1)
            alpha += rho * (node_degree.reshape(-1, 1) * x - C.dot(z))

            primal_gap.append(LA.norm(x - x_opt) / LA.norm(x_opt))
            primal_residual.append(LA.norm(A.dot(x) - B.dot(z)))
            dual_residual.append(LA.norm(rho * C.dot(z - z_prev)))

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
