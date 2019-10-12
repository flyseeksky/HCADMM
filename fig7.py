# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:09:20 2018

@author: maxxx971

For in-network acceleration, but on RANDOM graphs.

"""

#%% import libraries
from Simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#%%
# 1. graph
n_nodes = 50     # number of nodes
d = 3            # dimension of variable at each node
np.random.seed(1000)
# 2. function
# objective value
v = np.random.rand(n_nodes, d)
# optimal value
x_opt = v.mean()

# 3. simulation setting
graphs = [nx.lollipop_graph(n_nodes//2, n_nodes - n_nodes//2),
          nx.connected_caveman_graph(n_nodes//5, 5),
          Simulator.erdos_renyi(n_nodes, 0.05, seed=501),
          Simulator.erdos_renyi(n_nodes, 0.1, seed=1000)]
graph_name = ['Lollipop', 'Caveman', 'ER(p=0.05)', 'ER(p=0.1)']
line_style = ['--rd', '-rd',
              '--c^', '-c^',
              '--bs', '-bs',
              '--go', '-go']
best_penalty = [{'D-CADMM': 5, 'H-CADMM': 5.5},
                {'D-CADMM': 1.57, 'H-CADMM': 2.45},
                {'D-CADMM': 1.5, 'H-CADMM':1.85},
                {'D-CADMM': .75, 'H-CADMM': 1.4}]
all_mode = ['D-CADMM', 'H-CADMM']
max_iter = 500
epsilon = 1e-8
# start simulation
setting = {'penalty': -1, 'max_iter': max_iter, 'objective': v,
           'initial': 0 * np.random.randn(n_nodes, d),
#           'random_hyperedge': .5,
           'epsilon': epsilon}

#title_str = '{}, Nodes: {}, Edges: {}'.format(graph_type, n_nodes,
#             g.number_of_edges())

#%% simulation
sim_data = []
for G, name, rho in zip(graphs, graph_name, best_penalty):
    sim = Simulator(G, simulation_setting=setting)
    for mode in all_mode:
        data = {}
        sim.mode = mode
        sim.setting['penalty'] = rho[mode]
        opt_gap, primal_residual, dual_residual,_ = sim.run_least_squares()
        data['legend'] = name + ' ' + sim.mode
        data['opt_gap'] = opt_gap
        data['primal_residual'] = primal_residual
        data['dual_residual'] = dual_residual
        sim_data.append(data)


#%% plot
n_markers = 20
#marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)
marker_at = setting['max_iter'] // n_markers

# accuracy vs iteration
fig = plt.figure(1, figsize=(8, 6))
# fig = plt.figure()
for data, style in zip(sim_data, line_style):
    plt.semilogy(data['opt_gap'], style, label=data['legend'],
                 markevery=marker_at)

plt.ylabel('Accuracy')
plt.xlabel('Iterations/Communicatino cost')
# plt.title(title_str)
plt.ylim(ymin=epsilon)
plt.legend()


fig.tight_layout()
#from matplotlib2tikz import save as tikz_save
#tikz_save('in_net_rand.tex', figureheight='4cm', figurewidth='6cm')
plt.show()
