# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:56:14 2018

@author: maxxx971
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import GraphToolkit as gt


#n_values = [11, 21, 31, 41, 51, 61]
n_values = [11, 21]
ROW = 1
COL = len(n_values)

kappa = []
for n in n_values:
    graphs = [gt.ER(n, .6), nx.star_graph(n-1)]
#    graphs = [nx.star_graph(n-1), nx.complete_graph(n),
#              gt.ER(n, .05), gt.ER(n, .2), gt.ER(n, .3),
#              nx.cycle_graph(n-1), nx.path_graph(n)]
    k_col = []
    for G in graphs:
        k_col.append(gt.cond_ratio(G))
    kappa.append(k_col)
kappa = np.array(kappa)