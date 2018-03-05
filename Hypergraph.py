# hypergraph class
import numpy as np


class Hypergraph:

    def __init__(self, incidence, hyperedge=[], num=-1):
        assert isinstance(incidence, np.ndarray), 'Wrong type'
        if hyperedge:
            self.incidence = Hypergraph.acc(incidence, hyperedge)
        else:
            self.incidence = Hypergraph.in_network_acc(incidence, num=num)

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

    @staticmethod
    def in_network_acc(incidence, th=2, num=-1):
        C = incidence.copy() # incidence matrix
        N = C.shape[0]
        HC = []
        FC = []
        node_consider = np.full((N,), True)
        node_degree = np.sum(C, axis=1)
        node_label = np.arange(N)
        while np.any(node_consider)\
        and np.max(node_degree[node_consider]) >= th:
            max_node_index = np.argmax(node_degree[node_consider])
            sub_node_index = node_label[node_consider]
            node = sub_node_index[max_node_index]
            incident_edge = C[node, :]
            hyperedge = np.sum(C[:, incident_edge.astype(bool)], axis=1) > 0

            # rebuild incidence matrix
            HC.append(hyperedge)
            C = C[:, incident_edge == 0]

            # exclude node already in other hyperedges
            node_degree = np.sum(C, axis=1)
            node_consider = node_consider & ~hyperedge

            # check if enough FCs
            FC.append(node)
            if num > 0 and len(FC) >= num:
                break
        incident_edge = np.sum(C[node_consider, :], axis=0).astype(bool)
        HC.append(C[:, incident_edge])
        HC = np.column_stack(HC)
        return HC

    @staticmethod
    def acc(incidence, hyperedge):
        """
        Generate hypergraph with hyperedge specified by hyperedge
        For the moment, only one hyperedge is suppored.
        """
        C = incidence.copy()
        N, M = C.shape
        node_list = np.arange(N)

        # get edge index to build hyperedge
        edge_combine = np.full((M,), False)
        for edge in range(M):
            node1, node2 = node_list[C[:, edge].astype(bool)]
            if node1 in hyperedge and node2 in hyperedge:
                edge_combine[edge] = True

        # build hyperedge
        he = np.sum(C[:, edge_combine], axis=1).astype(bool).astype(float)
        # build incidence
        HC = np.column_stack((he, C[:, ~edge_combine]))
        return HC

