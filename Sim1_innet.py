# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:09:20 2018

@author: maxxx971

This is for simulation 1, which will produce two figure:
   1. accuracy vs iteration number;
   2. accuracy vs communication cost;

The main purpose of this simulation is to demonstrate the ability of
"in-network acceleration" for typical graphs, including:
   1. line graph, largest diameter with given number of nodes
   2. cycle graph, diameter shrink by half compared to line graph
   3. star graph, largest acceleration achieved
   4. complete graph, limited acceleration since diameter=2
   5. lollipop graph, show that even a single path can significantly reduce
      convergence speed of fully decentralized method
   6. Erdos-Renyi, no specific reasons, maybe not included
   7. grid graph, again acceleration exists, but not impressive
"""

#%% import libraries
from Simulator import Simulator
#from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#%% simulation setting
# 1. graph: N, d
# 2. function: objective value, opt_value
# 3. penalty parameters
# 4. simulation: max_iter, epsilon, initial value


# 1. graph
n_nodes = 50     # number of nodes
d = 3            # dimension of variable at each node

# 2. function
# objective value
v = np.random.rand(n_nodes, d)
# optimal value
x_opt = v.mean()

# 3. simulation setting
graphs = [nx.path_graph(n_nodes),
          nx.cycle_graph(n_nodes),
          nx.star_graph(n_nodes-1)]
graph_name = ['Line', 'Cycle', 'Star']
line_style = ['--rd', '-rd',
              '--bs', '-bs',
              '--go', '-go']
best_penalty = [{'D-CADMM': 5, 'H-CADMM': 5},# Line
                {'D-CADMM': 5, 'H-CADMM': 5},# Cycle
                {'D-CADMM': 5, 'H-CADMM': 5}]# Star
all_mode = ['D-CADMM', 'H-CADMM']
max_iter = 500
epsilon = 1e-8
# start simulation
setting = {'penalty': -1, 'max_iter': max_iter, 'objective': v,
           'initial': 0 * np.random.randn(n_nodes, d),
           'epsilon': epsilon}

#title_str = '{}, Nodes: {}, Edges: {}'.format(graph_type, n_nodes,
#             g.number_of_edges())

#%% simulation
# centralized
#sim.mode = 'centralized'
#sim.simulation_setting['penalty'] = best_penalty[0]
#c_opt_gap, c_primal_residual, c_dual_residual = sim.run_least_squares()

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
        data['primal_residual'] = primal_residual
        data['dual_residual'] = dual_residual
        sim_data.append(data)

#%% plot
n_markers = 20
#marker_at = np.array(range(0, setting['max_iter'],
#                           setting['max_iter'] // n_markers))
marker_at = setting['max_iter'] // n_markers
# accuracy vs iteration
fig = plt.figure(1, figsize=(8, 6))
# fig = plt.figure()
for data, style in zip(sim_data, line_style):
    plt.semilogy(data['opt_gap'], style, label=data['legend'],
                 markevery=marker_at)

plt.ylabel('Accuracy')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=epsilon)
plt.legend()


fig.tight_layout()
plt.show()
