# hypergraph class

import numpy as np


class Hypergraph:

    def __init__(self, incidence):
        assert isinstance(incidence, np.ndarray), 'Wrong type'
        self.incidence = incidence
        self.in_network_acc()

    def node_degree_matrix(self):
        degree = np.sum(self.incidence, axis=1)
        return np.diag(degree)

    def edge_degree_matrix(self):
        degree = np.sum(self.incidence, axis=0)
        return np.diag(degree)

    def incidence_matrix(self):
        return self.incidence

    def neighbor(self, node):
        C = self.incidence == 1
        N = C.shape[0]
        node_seq = np.arange(N)
        incident_edge = C[node, :]
        neighbor_index = np.sum(C[:, incident_edge], axis=1) >= 1
        return node_seq[neighbor_index]


    def in_network_acc(self, threshold=2, num=-1):
        C = self.incidence.copy() # incidence matrix
        HC = []
        node_consider = np.full((C.shape[0],), True)
        node_degree = np.sum(C, axis=1)
        while not np.alltrue(node_consider == False) and np.max(node_degree[node_consider]) >= threshold:
            max_node_index = np.argmax(node_degree[node_consider])
            sub_node_index = np.arange(C.shape[0])[node_consider]
            node = sub_node_index[max_node_index]
            edges_to_combine = C[node, :]
            hyperedge = np.sum(C[:, edges_to_combine.astype(bool)], axis=1) > 0

            # rebuild incidence matrix
            HC.append(hyperedge)
            C = C[:, edges_to_combine == 0]

            # exclude node already in other hyperedges
            node_degree = np.sum(C, axis=1)
            node_consider = node_consider & ~hyperedge
        edges_to_combine = np.sum(C[node_consider, :], axis=0).astype(bool)
        HC.append(C[:, edges_to_combine])
        HC = np.column_stack(HC)
        self.incidence = HC

