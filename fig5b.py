# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:09:20 2018

@author: maxxx971

General HCADMM simulation
"""

#%%
from Simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#%% simulation setting
# 1. graph
n_nodes = 50     # number of nodes
d = 1            # dimension of variable at each node
#np.random.seed(200)

# 2. objective value
np.random.seed(100)  # for reproductivity
v = np.random.rand(n_nodes, d)

all_mode = ['D-CADMM', 'H-CADMM']
max_iter = 500
epsilon = 1e-8
setting = {'penalty': -1,
           'max_iter': max_iter,
           'objective': v,
           'initial': 0 * np.random.randn(n_nodes, d),
           'random_hyperedge': .2,  # ratio of nodes in hyperedge
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
best_penalty = [{'D-CADMM': 5.9, 'H-CADMM': 2.7},
                {'D-CADMM': 1.55, 'H-CADMM': 1.528},
                {'D-CADMM': 1.15, 'H-CADMM': 1.175},
                {'D-CADMM': .7, 'H-CADMM': .7}]


#%% simulation

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
plt.xlim([-5, 500])
plt.tight_layout()
plt.legend()

#from matplotlib2tikz import save as tikz_save
#tikz_save('g_acc_iter2.tex', figureheight='4cm', figurewidth='6cm')

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
plt.xlim([-1000, 80000])
plt.legend()

fig.tight_layout()
#tikz_save('g_acc_comm2.tex', figureheight='4cm', figurewidth='6cm')
plt.show()
