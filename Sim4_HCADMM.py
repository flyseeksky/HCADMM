# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:09:20 2018

@author: maxxx971

General HCADMM simulation
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
d = 1            # dimension of variable at each node
np.random.seed(200)

# 2. function
# objective value
v = np.random.rand(n_nodes, d)

all_mode = ['D-CADMM', 'H-CADMM']
max_iter = 500
epsilon = 1e-8
# start simulation
setting = {'penalty': -1,
           'max_iter': max_iter,
           'objective': v,
           'initial': 0 * np.random.randn(n_nodes, d),
           'random_hyperedge': .2,  # ration of nodes in hyperedge
           'epsilon': epsilon,
           'n_FC': -1}

#%%
# 3. simulation setting
graphs = [nx.lollipop_graph(n_nodes//2, n_nodes - n_nodes//2 ),
          nx.connected_caveman_graph(n_nodes//5, 5),
          Simulator.erdos_renyi(n_nodes, 0.05),
          Simulator.erdos_renyi(n_nodes, 0.1, seed=1000)]
graph_name = ['Lollipop', 'Caveman', 'ER(p=0.05)', 'ER(p=0.1)']
line_style = ['--rd', '-rd',
              '--m^', '-m^',
              '--bs', '-bs',
              '--go', '-go']
best_penalty = [{'D-CADMM': 5.3, 'H-CADMM': 2.2},
                {'D-CADMM': 1.55, 'H-CADMM': 1.58},
                {'D-CADMM': 1.15, 'H-CADMM': 1.175},
                {'D-CADMM': .7, 'H-CADMM': .7}]

#%%
#graphs = [nx.path_graph(n_nodes),
#          nx.lollipop_graph(n_nodes//2, n_nodes - n_nodes//2 ),
#          nx.connected_caveman_graph(n_nodes//5, 5),
#          Simulator.erdos_renyi(n_nodes, 0.05)]
#graph_name = ['Line', 'Lollipop', 'Caveman', 'ER(p=0.05)']
#line_style = ['--rd', '-rd',
#              '--c^', '-c^',
#              '--bs', '-bs',
#              '--go', '-go']
#best_penalty = [{'D-CADMM': 5, 'H-CADMM': 5},
#                {'D-CADMM': 5, 'H-CADMM': 5},
#                {'D-CADMM': 5, 'H-CADMM': 5},
#                {'D-CADMM': 5, 'H-CADMM': 5}]

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
        opt_gap, primal_res, dual_res, edges = sim.run_least_squares()
        data['legend'] = name + ' ' + sim.mode
        data['opt_gap'] = opt_gap
        data['primal_residual'] = primal_res
        data['dual_residual'] = dual_res
        data['edges'] = edges
        sim_data.append(data)

#%% plot
n_markers = 20
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
plt.ylim(ymin=1e-8)
plt.legend()

#%%
# accuracy vs communication
fig = plt.figure(2, figsize=(8, 6))
for data, style in zip(sim_data, line_style):
    edges = data['edges']
    comm_cost = np.arange(len(data['opt_gap'])) * edges
    plt.semilogy(comm_cost, data['opt_gap'], style, label=data['legend'],
                 markevery=marker_at)

plt.xlabel('Communication cost')
plt.ylabel('Accuracy')
plt.ylim(ymin=1e-8)
plt.legend()

fig.tight_layout()
plt.show()
