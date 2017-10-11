from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Line graph with a large diameter needs more iterations to achieve certain accuracy
# simulation parameters
n_nodes = 500     # number of nodes
v = np.random.rand(n_nodes) * 10 + 1 * np.random.randn((n_nodes))
x_opt = v.mean()
setting = {'penalty': 1, 'max_iter': 200, 'objective': v, 'initial':np.random.randn(n_nodes)}

# generate graph
graph_type = 'Erdos Renyi'

if graph_type == 'Line Graph':
    g = nx.path_graph(n_nodes)
elif graph_type == 'Erdos Renyi':
    prob = .05
    g = Simulator.erdos_renyi(n_nodes, prob)
elif graph_type == 'Star Graph':
    g = nx.star_graph(n_nodes - 1)
elif graph_type == 'Cycle Graph':
    g = nx.cycle_graph(n_nodes)
else:
    raise Exception('Unsupported graph type')

# start simulation
sim = Simulator(g, simulation_setting=setting)

# centralized
sim.mode = 'centralized'
sim.simulation_setting['penalty'] = 1
c_opt_gap, c_primal_residual, c_dual_residual = sim.run_least_squares()

# hybrid
sim.mode = 'hybrid'
sim.simulation_setting['penalty'] = .2
h_opt_gap, h_primal_residual, h_dual_residual = sim.run_least_squares()

# decentralized ADMM
sim.mode = 'decentralized'
sim.simulation_setting['penalty'] = .05
d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()


marker_at = range(0, setting['max_iter'], setting['max_iter'] // 10)
title_str = '{}, N={}'.format(graph_type, n_nodes)
f1 = plt.figure(1)
plt.semilogy(d_opt_gap, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.semilogy(c_opt_gap, '-s', lw=2, label='centralized', markevery=marker_at)
plt.semilogy(h_opt_gap, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.ylabel('Relative Optimality gap $||x - x^\star||^2/||x^\star||^2$')
plt.xlabel('Iterations')
plt.title(title_str)
plt.ylim(ymin=1e-8)
plt.legend()


# plt.figure(2)
# plt.semilogy(d_primal_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
# plt.semilogy(c_primal_residual, '-s', lw=2, label='centralized', markevery=marker_at)
# plt.semilogy(h_primal_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
# plt.title(title_str)
# plt.xlabel('Iterations')
# plt.ylabel('Primal residual')
# plt.ylim(ymin=1e-8)
# plt.legend()
#
# plt.figure(3)
# plt.semilogy(d_dual_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
# plt.semilogy(c_dual_residual, '-s', lw=2, label='centralized', markevery=marker_at)
# plt.semilogy(h_dual_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
# plt.title(title_str)
# plt.xlabel('Iterations')
# plt.ylabel('Dual residual')
# plt.ylim(ymin=1e-8)
# plt.legend()

f1.savefig(graph_type + '.pdf', bbox_inches='tight')
plt.show()
