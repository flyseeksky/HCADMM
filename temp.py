# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:20:54 2018

@author: maxxx971
"""

#import numpy as np
#from scipy import linalg
#import matplotlib.pyplot as plt
import networkx as nx
import GraphToolkit as gt



P = nx.star_graph(81)
#hyeredge = gt.greedy_hyperedge(P)
#print(hyeredge)
C = gt.get_H_incidence(P)
#print(C)

r = gt.cond_ratio(P)
print(r)