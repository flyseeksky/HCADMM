# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:56:14 2018

@author: maxxx971
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import GraphToolkit as gt


n_values = [11, 21, 31, 41, 51]
ROW = 4
COL = len(n_values)

k = np.zeros((ROW, COL))
for col, n in enumerate(n_values):
    graphs = [nx.path_graph(n), nx.cycle_graph(n), nx.star_graph(n-1),
          nx.complete_graph(n)]
    for row, G in enumerate(graphs):
        k[row, col] = gt.cond_ratio(G)