# -*- coding: utf-8 -*-
"""
GraphTookit

Created on Sat Feb 24 17:01:33 2018

@author: maxxx971
"""

import networkx as nx
import numpy as np
from scipy import linalg
from operator import itemgetter
from Hypergraph import Hypergraph
import scipy.sparse as sps



def cond_ratio(G):
    Cd = get_D_incidence(G)
    Ch = get_H_incidence(G)
    Kh = cond_num(Ch)
    Kd = cond_num(Cd)
    return Kd / Kh


def cond_num(C):
    """
    Compute condition number of graph.
    C is the incidence matrix.
    """
    N = C.shape[0]
    Einv = sps.diags(1 / np.squeeze(np.asarray(C.sum(axis=0))))
    D = sps.diags(np.squeeze(np.asarray(C.sum(axis=1))))
    M1 = C.dot(Einv).dot(C.T)
    M2 = D - M1

    L = linalg.eigh(M1.toarray(), eigvals_only=True, eigvals=(N-1, N-1))[0]
    l = linalg.eigh(M2.toarray(), eigvals_only=True, eigvals=(1, 1))[0]

    return L, l


def get_D_incidence(G):
    return np.asarray(nx.incidence_matrix(G).todense())


def get_H_incidence(G):
    Cd = np.asarray(nx.incidence_matrix(G).todense())
    H = Hypergraph(Cd)
    return H.incidence_matrix()

def ER(n, p):
    """
    Create a connected Erdos Renyi graph (n, p)
    """
    G = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)
#        G = nx.random_geometric_graph(n, p)
#        G = nx.gnm_random_graph(n, (1+p)*n)
    return G

#def greedy_hyperedge(G, threshold=4):
#    """
#    Automatically find patterns to build a hybrid model.
#    """
#    assert isinstance(G, nx.Graph), 'Graph must'
#    'be instance of networkx Graph'
#
#    # NOW FCs can connect with each other
#    degree_dict = dict(G.degree)
#    node_degree_list = sorted(degree_dict.items(), key=lambda x: x[1],
#                              reverse=True)
#    all_edges = set(G.edges)
#    FC = set()
#
#    while degree_dict:
#        # current node to consider: highest degree
#        node, degree = node_degree_list[0]
#        # find its neighbors in current hypergraph including itself
#        neighbor = set(nx.neighbors(G, node))
#        neighbor.difference_update(FC)  # exclue FCs
#        if len(neighbor) < threshold:
#            break
#        if neighbor:
#            neighbor.add(node)
#            ###### update node_degree_list
#            # remove current node
#            edges_to_remove = set()
#            for n in neighbor:
#                n_neighbor = set(nx.neighbors(G, n))
#                # for hyperedge in hyperedge_list:
#                #     if n in hyperedge:
#                #         n_neighbor.difference_update(hyperedge)
#                common_neighbor = n_neighbor & neighbor
#                # udpate degree
#                # k = len(common_neighbor)
#                # if k > 0:
#                #     degree_dict[n] -= k - 1
#                # all edges incident to node n
#                e1 = {(n, n1) for n1 in common_neighbor if n <= n1}
#                e2 = {(n1, n) for n1 in common_neighbor if n1 <= n}
#                edges_to_remove.update(e1 | e2)
#                edge_no_before = len(all_edges)
#                all_edges.difference_update(edges_to_remove)
#                edge_no_after = len(all_edges)
#                if edge_no_before - edge_no_after >= 2:
#                    degree_dict[n] -= edge_no_before - edge_no_after - 1  # hyperedge
#            # add hyperedge into edge list
#            all_edges.add(tuple(neighbor))
#        del degree_dict[node]
#        FC.add(node)
#        node_degree_list = sorted(degree_dict.items(), key=lambda x: x[1],
#                                  reverse=True)
#
#    # hyperedge_list += list(all_edges)
#    return all_edges




