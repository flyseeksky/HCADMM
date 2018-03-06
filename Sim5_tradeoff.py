# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:09:20 2018

@author: maxxx971

This is for simulation 3, which will produce two figure:
   1. accuracy vs iteration number;
   2. accuracy vs communication cost;

The main purpose of this simulation is to demonstrate the ability of
general acceleration for typical graphs, including:
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
np.random.seed(200)
v = np.random.rand(n_nodes, d)
# optimal value
x_opt = v.mean()

# 3. simulation setting
graphs = [nx.path_graph(n_nodes),
          nx.cycle_graph(n_nodes),
          nx.lollipop_graph(n_nodes//2, n_nodes - n_nodes//2 ),
          Simulator.erdos_renyi(n_nodes, 0.1)]
graph_name = ['Line', 'Cycle', 'Lollipop', 'ER(p=0.1)']
line_style = ['-rd',
              '-c^',
              '-bs',
              '-go']
best_penalty = [7.5, 3.7, 5.56, 1.4]
mode = 'H-CADMM'
max_iter = 1000
# start simulation
setting = {'penalty': -1, 'max_iter': max_iter, 'objective': v,
           'initial': 0 * np.random.randn(n_nodes, d),
           'epsilon': 1e-8,
#           'random_hyperedge': [],
           'n_FC': -1}
n_FC = [1, 5, 10, 15, 20, 25]

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
    sim.mode = mode
    sim.setting['penalty'] = rho
    data = {}
    data['legend'] = name
    n_iter = np.zeros_like(n_FC)
    n_edge = np.zeros_like(n_FC)
    for i, nfc in enumerate(n_FC):
        sim.setting['n_FC'] = nfc
        # sim.setting['penalty'] = rho[mode]
        opt_gap, primal_res, dual_res, edges = sim.run_least_squares()
        n_iter[i] = len(opt_gap)
        n_edge[i] = edges
    data['n_iter'] = n_iter
    data['n_edge'] = n_edge
    sim_data.append(data)

#%%
#plt.figure()
#plt.semilogy(sim_data[0]['opt_gap'], '--r', label='frist')
#plt.semilogy(sim_data[1]['opt_gap'], '-b', label='second')
#
#plt.legend()
#plt.show()
## decentralized ADMM
#   sim.mode = 'decentralized'
#   sim.simulation_setting['penalty'] = best_penalty[2]
#   d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()


#%% plot
n_markers = 20
marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)

# accuracy vs iteration
fig = plt.figure(1, figsize=(8, 6))
# fig = plt.figure()
for data, style in zip(sim_data, line_style):
    plt.plot(n_FC, data['n_iter'], style, label=data['legend'])
#                 markevery=marker_at)

plt.ylabel('Iterations needed')
plt.xlabel('Number of local FCs')
# plt.title(title_str)
#plt.ylim(ymin=1e-8)
plt.legend()

#%%
# accuracy vs communication
#fig = plt.figure(2, figsize=(8, 6))
#for data, style in zip(sim_data, line_style):
#    edges = data['edges']
#    comm_cost = np.arange(len(data['opt_gap'])) * edges
#    plt.semilogy(comm_cost, data['opt_gap'], style, label=data['legend'],
#                 markevery=marker_at)
#
#plt.xlabel('Communication cost')
#plt.ylabel('Accuracy')
#plt.ylim(ymin=1e-8)
#plt.legend()
#
#fig.tight_layout()
#plt.show()
