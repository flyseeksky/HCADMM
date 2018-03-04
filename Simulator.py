#!/usr/local/bin/python3
from operator import itemgetter
import scipy.sparse as sps
import numpy as np
import scipy.linalg as LA
import networkx as nx
from Hypergraph import Hypergraph
from GraphToolkit import cond_num
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Simulator():
    def __init__(self, graph, mode='', simulation_setting=None):
        assert isinstance(graph, nx.Graph), 'graph must be'
        ' a networkx Graph object'
        self.graph = graph
        self.mode = mode
        self.setting = simulation_setting

    def run_least_squares(self):
        logger.info('===== Mode: ' + self.mode + ' =====')

        if self.mode == 'C-CADMM':
            C = np.ones((self.graph.number_of_nodes(), 1))
        elif self.mode == 'D-CADMM':
            C = nx.incidence_matrix(self.graph)
            logger.debug("Hybrid C = \n" + str(C.todense()))
        elif self.mode == 'H-CADMM':
            C = np.asarray(nx.incidence_matrix(self.graph).todense())
            H = Hypergraph(C)
            C = H.incidence_matrix()
            logger.debug("Hybrid C = \n" + str(C))
        else:
            raise ValueError("mode must be set")
        # check to see if penalty need to be set
        if self.setting['penalty'] <= 0:
            # get L and l
            L, l = cond_num(C)
            self.setting['penalty'] = np.sqrt(2 / (L*l*(1 + 2*L/l)))
            logger.debug('penalty = ' + str(self.setting['penalty']))

        # convert C to A, B to compute residual
        A, B = self.incidence_to_ab(C)

        # reshape them as (-1, 1), which is a 2D vector
        node_degree, edge_degree = (np.squeeze(np.asarray(C.sum(axis=1))),
                                    np.squeeze(np.asarray(C.sum(axis=0))))

        # extract simulation settings
        setting = self.setting
        rho, v, x0, max_iter = (setting['penalty'], setting['objective'],
                                setting['initial'], setting['max_iter'])
        x_opt = v.mean(axis=0)

        # initial value
        z0 = C.T.dot(x0) / edge_degree.reshape(-1, 1)
        alpha0 = np.zeros_like(x0)
        primal_gap, primal_residual, dual_residual = [], [], []

        logger.info('Mode: %s, starting for-loop', self.mode)
        x, z, alpha = x0, z0, alpha0
        for i in range(max_iter):
            z_prev = z  # save for computing dual residual
            x = (v - alpha + rho * C.dot(z)) / (1 +
                rho*node_degree.reshape(-1, 1))
            z = C.T.dot(x) / edge_degree.reshape(-1, 1)
            alpha += rho * (node_degree.reshape(-1, 1) * x - C.dot(z))

            primal_gap.append(LA.norm(x - x_opt) / LA.norm(x_opt))
            primal_residual.append(LA.norm(A.dot(x) - B.dot(z)))
            dual_residual.append(LA.norm(rho * C.dot(z - z_prev)))

            # debug printing
            step = max_iter // 10
            if i % step == step - 1:
                logger.info('Progress %.1f', 100 * (i+1)/ max_iter)

        logger.info('Mode: %s, ending for loop', self.mode)
        return primal_gap, primal_residual, dual_residual

    @staticmethod
    def erdos_renyi(n_nodes, prob, seed=None):
        """
        Randomly generate a connected graph using Erdos-Renyi model
        """
        G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n_nodes, prob, seed=seed)
        return G

    @staticmethod
    def incidence_to_ab(incidence):
        """
        Generate matrix A,B from incidence matrix
        """
        assert isinstance(incidence, np.ndarray) \
        or sps.isspmatrix(incidence), 'Invalid incidence matrix'

        # Convert incidence matrix to sparse format
        if not sps.isspmatrix(incidence):
            sp_incidence = sps.csr_matrix(incidence)
        else:
            sp_incidence = incidence

        n, m = sp_incidence.shape
        t = sp_incidence.nnz

        # Elements of A, B need to be changed, so lil_matrix is better
        A, B = sps.lil_matrix((t, n)), sps.lil_matrix((t, m))
        row_index, col_index = sp_incidence.nonzero()
        for idx, (row, col) in enumerate(zip(row_index, col_index)):
            A[idx, row] = 1
            B[idx, col] = 1
        return A.tocsr(), B.tocsr()
