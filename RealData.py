from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Read Power Grid network
path = './dataset/power.gml'
g = nx.read_gml(path, label='id')
title_str = 'US Power Grid, Nodes: {}, Edges: {}'.format(g.order(), g.number_of_edges())

# karate club network
# g = nx.karate_club_graph()
# best penalty: (1, 1, .6)
# title_str = 'Karate Club, Nodes: {}, Edges: {}'.format(g.order(), g.number_of_edges())

# simulation parameters
n_nodes = g.order()                     # number of nodes

v = np.random.rand(n_nodes) * 10 + 10
x_opt = v.mean()

setting = {'penalty': 1, 'max_iter': 1000, 'objective': v, 'initial': np.random.randn(n_nodes)}


# start simulation
sim = Simulator(g, simulation_setting=setting)

# centralized
sim.mode = 'centralized'
sim.simulation_setting['penalty'] = 1
c_opt_gap, c_primal_residual, c_dual_residual = sim.run_least_squares()

# hybrid
sim.mode = 'hybrid'
sim.simulation_setting['penalty'] = 15
h_opt_gap, h_primal_residual, h_dual_residual = sim.run_least_squares()

# decentralized ADMM
sim.mode = 'decentralized'
sim.simulation_setting['penalty'] = 16
d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()

#plotting figures
marker_at = range(0, setting['max_iter'], setting['max_iter'] // 20)
plt.figure(1)
plt.semilogy(d_opt_gap, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.semilogy(c_opt_gap, '-s', lw=2, label='centralized', markevery=marker_at)
plt.semilogy(h_opt_gap, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.ylim(ymin=1e-8)
plt.ylabel('Relative Optimality gap $||x - x^\star||^2/||x^\star||^2$')
plt.xlabel('Iterations')
plt.title(title_str)
plt.grid()
plt.legend()


# plt.figure(2)
# plt.semilogy(d_primal_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
# plt.semilogy(c_primal_residual, '-s', lw=2, label='centralized', markevery=marker_at)
# plt.semilogy(h_primal_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
# plt.ylim(ymin=1e-8)
# plt.title(title_str)
# plt.ylim(ymin=1e-8)
# plt.xlabel('Iterations')
# plt.ylabel('Primal residual')
# plt.legend()
#
# plt.figure(3)
# plt.semilogy(d_dual_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
# plt.semilogy(c_dual_residual, '-s', lw=2, label='centralized', markevery=marker_at)
# plt.semilogy(h_dual_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
# plt.ylim(ymin=1e-8)
# plt.title(title_str)
# plt.ylim(ymin=1e-8)
# plt.xlabel('Iterations')
# plt.ylabel('Dual residual')
# plt.legend()

# plt.figure()
# nx.draw(g)

plt.show()
