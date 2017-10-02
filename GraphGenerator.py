import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def check_hyper_edges(incidence, hyper_edges):
    assert isinstance(incidence, np.ndarray)
    assert isinstance(hyper_edges, list)
    # verify total number of nodes in hyperedges are consistent
    n_edges = incidence.shape[1]
    total_edges = 0
    for edge in hyper_edges:
        if isinstance(edge, list):
            total_edges += len(edge)
        else:
            total_edges += 1
    assert total_edges == n_edges


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


def random_connected_graph(n_nodes, prob):
    """
    randomly generate a connected graph using Erdos-Renyi model
    :param n_nodes: number of nodes
    :param prob: the probability of an edge
    :return: an Networkx object
    """

    G = nx.erdos_renyi_graph(n_nodes, prob)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_nodes, prob)

    return G


def hyper_incidence(incidence, hyper_edges):
    """
    This function convert an incidence matrix of a simple graph to an incidence of hypergraph, whose
    edges specified by parameter hyper_edges
    :param incidence: a matrix
    :param hyper_edges: a list of hyper edges, each of which is a list of nodes
    :return:
    """
    check_hyper_edges(incidence, hyper_edges)
    hyper_incidence = np.zeros((incidence.shape[0], len(hyper_edges)), dtype=np.int)
    for n, edge in enumerate(hyper_edges):  # assuming edges are ascending ordered
        if isinstance(edge, list) and len(edge) > 1: # if edge is a list, sub_incidence is a ndarray
            sub_incidence = incidence[:, edge]
            sub_hyper_incidence = sub_incidence.sum(axis=1) > 0
        else:                                        # if edge is a single number, sub_incidence is a vector
            sub_hyper_incidence = incidence[:, edge]
        hyper_incidence[:, n] = sub_hyper_incidence.astype(np.int).ravel()
    return hyper_incidence
