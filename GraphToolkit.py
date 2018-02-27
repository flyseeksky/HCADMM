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


def cond_ratio(G):
    Cd = get_D_incidence(G)
    Ch = get_H_incidence(G)
    Kd = cond_num(Cd)
    Kh = cond_num(Ch)
    return Kd / Kh


def cond_num(C):
    """
    Compute condition number of graph.
    C is the incidence matrix.
    """
    N = C.shape[0]
    Einv = np.diag(1 / C.sum(axis=0))
    D = np.diag(C.sum(axis=1))
    M1 = C.dot(Einv).dot(C.T)
    M2 = D - M1
    
    L = linalg.eigh(M1, eigvals_only=True, eigvals=(N-1, N-1))[0]
    l = linalg.eigh(M2, eigvals_only=True, eigvals=(1, 1))[0]
    
    return L / l


def get_D_incidence(G):
    return np.asarray(nx.incidence_matrix(G).todense())


def get_H_incidence(G):
    edge_list = greedy_hyperedge(G)
    M = len(edge_list)
    N = G.number_of_nodes()
    C = np.zeros((N, M))
    for idx, edge in enumerate(edge_list):
        C[edge, idx] = 1
    
    return C


def greedy_hyperedge(G, threshold=2):
    """
    Automatically find patterns to build a hybrid model.
    """
    assert isinstance(G, nx.Graph), 'Graph must'
    'be instance of networkx Graph'
    
    threshold = max(dict(G.degree).values())

    node_degree_list = sorted(list(G.degree), 
                              key=itemgetter(1), reverse=True)
    # TODO fix number of local centers
    # consider according to descending degree order
    node_degree_list_to_consider = [nd for nd in node_degree_list 
                                    if nd[1] >= threshold]
    qualified_node_set = set([nd[0] for nd in node_degree_list_to_consider])
    all_edge_set = set(G.edges)

    hyper_edge_list = []
    remaining_edge_set = all_edge_set.copy()
    for node, _ in node_degree_list_to_consider:
        # if current node not qualify, next
        if node not in qualified_node_set:
            continue

        all_neighbors = tuple(sorted(nx.all_neighbors(G, node)))

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
    # remove edge cross well connected hyperedges
    remove_edge = set()
    h_index1, h_index2 = np.nan, np.nan
    for edge in remaining_edge_set:
        for idx, hyperedge in enumerate(hyper_edge_list):
            if not np.isnan(h_index1) and edge[0] in hyperedge:
                h_index1 = idx
            elif not np.isnan(h_index2) and edge[1] in hyperedge:
                h_index2 = idx
            if not np.isnan(h_index1 * h_index2):
                break
        if hyper_edge_list[h_index1].intersection(hyper_edge_list[h_index2]):
            remove_edge.update(edge)
    remaining_edge_set.difference_update(remove_edge)
    
    hyper_edge_list += list(remaining_edge_set)
    hyper_edge_list = sorted(hyper_edge_list)
    return hyper_edge_list


def ER(n, p):
    """
    Create a connected Erdos Renyi graph (n, p)
    """
    G = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)
    return G

