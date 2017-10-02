import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def check_hyper_edges(simple_adj, hyper_edges):
    assert isinstance(simple_adj, np.ndarray)
    assert isinstance(hyper_edges, list)
    # verify total number of nodes in hyperedges are consistent
    n_edges = simple_adj.shape[1]
    total_edges = 0
    for edge in hyper_edges:
        if isinstance(edge, list):
            total_edges += len(edge)
        else:
            total_edges += 1
    assert total_edges == n_edges


def node_to_edge_list(simple_adj, node_list):
    """
    Convert hyper identifiers from node list to edge list
    :param simple_adj: adjacency matrix for simple graph
    :param node_list: nodes list of each hyper edge
    :return:
    """
    edge_list = []
    adj = np.array(simple_adj)
    all_edges = np.arange(adj.shape[1])
    for nodes in node_list:
        sub_adj = adj[nodes, :]
        edge_idx = sub_adj.sum(axis=0) > 1
        edge_list.append(list(all_edges[edge_idx]))
    return edge_list


def random_connected_graph(n_nodes, prob):
    """
    randomly generate a graph using Erdos-Renyi model
    :param n_nodes: number of nodes
    :param prob: the probability of an edge
    :return: an Networkx object
    """

    G = nx.erdos_renyi_graph(n_nodes, prob)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_nodes, prob)

    return G


def simple_adj_to_hyper_adj(simple_adj, hyper_edges):
    """
    This function convert an adjacency matrix of a simple graph to an adjacency of hypergraph, whose
    edges specified by parameter hyper_edges
    :param simple_adj: a matrix
    :param hyper_edges: a list of hyper edges, each of which is a list of nodes
    :return:
    """
    check_hyper_edges(simple_adj, hyper_edges)
    hyper_adj = np.zeros((simple_adj.shape[0], len(hyper_edges)), dtype=np.int)
    for n, edge in enumerate(hyper_edges):
        if isinstance(edge, list) and len(edge) > 1: # if edge is a list, sub_adj is a ndarray
            sub_adj = simple_adj[:, edge]
            sub_hyper_adj = sub_adj.sum(axis=1) > 0
        else:                                        # if edge is a single number, sub_adj is a vector
            sub_hyper_adj = simple_adj[:, edge]
        hyper_adj[:, n] = sub_hyper_adj.astype(np.int).ravel()
    return hyper_adj
