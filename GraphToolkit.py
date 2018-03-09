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
    Lh, lh = cond_num(Ch)
    Kd, ld = cond_num(Cd)
    Kh, Kd = Lh/lh, Kd/ld
    return Kd / Kh


def cond_num(C):
    """
    Compute condition number of graph.
    C is the incidence matrix.
    """
    N = C.shape[0]
    if sps.issparse(C):
        C = C.toarray()
    Einv = np.diag(1 / C.sum(axis=0))
    D = np.diag(C.sum(axis=1))
    M1 = C.dot(Einv).dot(C.T)
    M2 = D - M1

    L = linalg.eigh(M1, eigvals_only=True, eigvals=(N-1, N-1))[0]
    l = linalg.eigh(M2, eigvals_only=True, eigvals=(1, 1))[0]

    return L, l


def get_D_incidence(G):
    return nx.incidence_matrix(G)


def get_H_incidence(G):
    Cd = nx.incidence_matrix(G)
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
