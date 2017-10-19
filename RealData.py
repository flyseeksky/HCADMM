from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Read Power Grid network
# best_penalty = (1, 15, 16)
# max_iter = 1400
# path = './dataset/power.gml'
# g = nx.read_gml(path, label='id')
# title_str = 'US Power Grid, Nodes: {}, Edges: {}'.format(g.order(), g.number_of_edges())
# filename = 'US_power_grid.pdf'

# karate club network
g = nx.karate_club_graph()
max_iter = 80
best_penalty = (1, 1, .6)
title_str = 'Karate Club, Nodes: {}, Edges: {}'.format(g.order(), g.number_of_edges())
filename = 'karate_club.pdf'

# simulation parameters
n_nodes = g.order()                     # number of nodes
np.random.seed(1)
v = .1 * np.random.randn(n_nodes) + np.random.randint(1, 10)
x_opt = v.mean()

setting = {'penalty': 1, 'max_iter': max_iter, 'objective': v, 'initial': np.random.randn(n_nodes)}


# start simulation
sim = Simulator(g, simulation_setting=setting)

# centralized
sim.mode = 'centralized'
sim.simulation_setting['penalty'] = best_penalty[0]
c_opt_gap, c_primal_residual, c_dual_residual = sim.run_least_squares()

# hybrid
sim.mode = 'hybrid'
sim.simulation_setting['penalty'] = best_penalty[1]
h_opt_gap, h_primal_residual, h_dual_residual = sim.run_least_squares()

# decentralized ADMM
sim.mode = 'decentralized'
sim.simulation_setting['penalty'] = best_penalty[2]
d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()

#plotting figures
marker_at = range(0, setting['max_iter'], setting['max_iter'] // 20)
fig = plt.figure(1, figsize=(8,6))
plt.semilogy(d_opt_gap, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.semilogy(c_opt_gap, '-s', lw=2, label='centralized', markevery=marker_at)
plt.semilogy(h_opt_gap, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.ylim(ymin=1e-5)
plt.ylabel('Relative accuracy $||x - x^\star||/||x^\star||$')
plt.xlabel('Iterations')
# plt.title(title_str)
# plt.grid()
plt.legend()

fig.tight_layout()
fig.savefig(filename, bbox='tight', pad_inches=0)


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
