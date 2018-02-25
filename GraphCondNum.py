# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:31:59 2018

@author: maxxx971
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg


def condition_number(C):
    """
    compute condition number of graph
    C is the incidence matrix of size N by M
    N is number of nodes and M is number of edges
    """
    D = np.diag(C.sum(axis=1))
    Einv = np.diag(1/C.sum(axis=0))
    M1 = C.dot(Einv).dot(C.T)
    M2 = D - M1

    # find the largest and smallest eigvalues
    N = C.shape[0]
    Lbd = linalg.eigh(M1, eigvals_only=True, eigvals=(N-1, N-1))[0]
    lbd = linalg.eigh(M2, eigvals_only=True, eigvals=(1, 1))[0]

    return Lbd / lbd

# number of nodes of path graph
N = 11

G = nx.path_graph(N)
C = np.asarray(nx.incidence_matrix(G).todense())
Einv = np.diag(1 / C.sum(axis=0))
D = np.diag(C.sum(axis=1))
M1 = C.dot(Einv).dot(C.T)
M2 = D - M1

L = linalg.eigh(M1, eigvals_only=True, eigvals=(N-1, N-1))[0]
l = linalg.eigh(M2, eigvals_only=True, eigvals=(1, 1))[0]
Rd = L / l
print('D-ratio', Rd)

# H-CADMM
M = int((N-1) / 2)  # number of edges

e_i = np.zeros((N,))
e_i[0:3] = 1
Ch = np.zeros((N, M))
for i in range(M):
    Ch[:, i] = e_i
    e_i = np.roll(e_i, 2)

Ehinv = np.diag(1 / Ch.sum(axis=0))
Dh = np.diag(Ch.sum(axis=1))
Mh1 = Ch.dot(Ehinv).dot(Ch.T)
Mh2 = Dh - Mh1

Lh = linalg.eigh(Mh1, eigvals_only=True, eigvals=(N-1, N-1))[0]
lh = linalg.eigh(Mh2, eigvals_only=True, eigvals=(1, 1))[0]

Rh = Lh / lh
print('H-ratio', Rh)

print('Acceleration: ', Rd / Rh)

M2 *= 2
Mh2 *= 3
#
#print("l / lh: ", l/lh)
#print("L / Lh: ", L/Lh)
#l_diff = np.abs(lh * .5 - l)
#L_diff = np.abs(Lh*1.2 - L)
#print('l relate to n', l_diff * N)
#print('L relate to n', L_diff * N)

def f(k):
    return 1 - np.cos(np.pi * k / N)
#################################################################
# line graph
# D-CADMM
#G = nx.path_graph(N)
#C = np.asarray(nx.incidence_matrix(G).todense())
#k_1 = condition_number(C)
#
## H-CADMM
#M = int((N-1) / 2)  # number of edges
#
#e_i = np.zeros((N,))
#e_i[0:3] = 1
#Ch = np.zeros((N, M))
#for i in range(M):
#    Ch[:, i] = e_i
#    e_i = np.roll(e_i, 2)
#
#k_2 = condition_number(Ch)
#
#print('\n=======================')
#print('Line Graph')
#print('D-CADMM: ', k_1)
#print('H-CADMM: ', k_2)
#print('Acceleration ratio: ', k_1 / k_2)
#print('=======================\n')

#################################################################
# star graph
#G = nx.star_graph(N-1)
#
## D-CADMM
#C = np.asarray(nx.incidence_matrix(G).todense())
#k_1 = condition_number(C)
#
## H-CADMM
#Ch = np.ones((N-1,1))
#k_2 = condition_number(Ch)
#
#print('\n=======================')
#print('Star Graph')
#print('D-CADMM: ', k_1)
#print('H-CADMM: ', k_2)
#print('Acceleration ratio: ', k_1 / k_2)
#print('=======================\n')

#################################################################
# cycle graph
# D-CADMM
#Nc = 9 + 1
#G = nx.cycle_graph(Nc)
#C = np.asarray(nx.incidence_matrix(G).todense())
#k_1 = condition_number(C)
#
## H-CADMM
#M = int(Nc / 2)  # number of edges
#
#e_i = np.zeros((Nc,))
#e_i[0:3] = 1
#Ch = np.zeros((Nc, M))
#for i in range(M):
#    Ch[:, i] = e_i
#    e_i = np.roll(e_i, 2)
#
#k_2 = condition_number(Ch)
#
#print('\n=======================')
#print('Cycle Graph')
#print('D-CADMM: ', k_1)
#print('H-CADMM: ', k_2)
#print('Acceleration ratio: ', k_1 / k_2)
#print('=======================\n')
#
##################################################################
## complete graph
## D-CADMM
#G = nx.complete_graph(N)
#C = np.asarray(nx.incidence_matrix(G).todense())
#k_1 = condition_number(C)
#
## H-CADMM
#Ch = np.ones((N, 1))
#k_2 = condition_number(Ch)
#
#print('\n=======================')
#print('Complete Graph')
#print('D-CADMM: ', k_1)
#print('H-CADMM: ', k_2)
#print('Acceleration ratio: ', k_1 / k_2)
#print('=======================\n')