# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:09:20 2018

@author: Meng Ma (maxxx971@umn.edu)

This is for simulation 1, which will produce two figure:
   1. accuracy vs iteration number;
   2. accuracy vs communication cost;
"""

#%%
from Simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#%% simulation setting
# 1. graph
n_nodes = 50     # number of nodes
d = 3            # dimension of variable at each node

# 2. function: objective value
v = np.random.rand(n_nodes, d)

# 3. simulation setting
graphs = [nx.path_graph(n_nodes),
          nx.cycle_graph(n_nodes),
          nx.star_graph(n_nodes-1)]
graph_name = ['Line', 'Cycle', 'Star']
line_style = ['--rd', '-rd',
              '--bs', '-bs',
              '--go', '-go']
# penalty parameters are hand-tuned, it would be better if there is some
# automatic way of finding the best parameters
best_penalty = [{'D-CADMM': 8.25, 'H-CADMM': 6.68},# Line
                {'D-CADMM': 4.03, 'H-CADMM': 3.3},# Cycle
                {'D-CADMM': 1, 'H-CADMM': 1}]# Star
all_mode = ['D-CADMM', 'H-CADMM']
max_iter = 500
epsilon = 1e-8

setting = {'penalty': -1,
           'max_iter': max_iter,
           'objective': v,
           'initial': np.zeros((n_nodes, d)),
           'epsilon': epsilon}

#%% simulation
sim_data = []
for G, name, rho in zip(graphs, graph_name, best_penalty):
    sim = Simulator(G, simulation_setting=setting)
    for mode in all_mode:
        data = {}
        sim.mode = mode
        sim.setting['penalty'] = rho[mode]
        opt_gap, primal_residual, dual_residual, _ = sim.run_least_squares()
        data['legend'] = name + ' ' + sim.mode
        data['opt_gap'] = opt_gap
        sim_data.append(data)

#%% plot
n_markers = 30
marker_at = setting['max_iter'] // n_markers
fig = plt.figure(1, figsize=(8, 6))
for data, style in zip(sim_data, line_style):
    plt.semilogy(data['opt_gap'], style, label=data['legend'],
                 markevery=marker_at)

plt.ylabel('Accuracy')
plt.xlabel('Iterations/Communication cost')
plt.ylim(ymin=epsilon)
plt.legend()
fig.tight_layout()


# uncomment the lines below to save figure to tikz file
#from matplotlib2tikz import save as tikz_save
#tikz_save('in_net_fixed.tex', figureheight='4cm', figurewidth='6cm')
plt.show()
