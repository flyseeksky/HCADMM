# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:56:14 2018

@author: maxxx971
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import GraphToolkit as gt


n = 11
graphs = [nx.path_graph(n), nx.cycle_graph(n), nx.star_graph(n-1),
          nx.complete_graph(n)]