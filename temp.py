# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:20:54 2018

@author: maxxx971
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import networkx as nx
import GraphToolkit as gt
from Hypergraph import Hypergraph



#P = nx.star_graph(81)
##hyeredge = gt.greedy_hyperedge(P)
##print(hyeredge)
#C = gt.get_H_incidence(P)
##print(C)
#
#r = gt.cond_ratio(graphs[3])
#print(r)
#print(r)
# G = nx.gnp_random_graph(8, 0.7, seed=1000)
# nx.draw_networkx(G)
# plt.show()
# print(gt.cond_ratio(G))

# G = nx.from_numpy_array()
# A = np.asarray(nx.incidence_matrix(G).todense())
#A = np.array([[1,0,0,0,0],
#              [1,1,1,0,0],
#              [0,1,0,0,0],
#              [0,0,1,1,0],
#              [0,0,0,1,1],
#              [0,0,0,0,1]])
#H = hg.Hypergraph(A)
#print(H.hyperincidence_matrix())
#%%
#G = nx.path_graph(10)
#C = nx.incidence_matrix(G).toarray()
#H = Hypergraph(C, num=4)
#Ch = H.incidence_matrix()
#print(Ch)
#
##%%
#G = nx.complete_graph(5)
#C = nx.incidence_matrix(G).toarray()
#H = Hypergraph(C, [1,2,3])
#Ch = H.incidence_matrix()
#print(Ch)

globaltemperatures = [14.4, 14.5, 14.8, 15.2, 15.5, 15.8]
numpirates = [45000, 20000, 15000, 5000, 400, 17]

fig = plt.figure(1, figsize=(8, 6))
plt.plot(numpirates, globaltemperatures, label="line")
plt.scatter(numpirates, globaltemperatures)
plt.show()