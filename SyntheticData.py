from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# simulation parameters
n_nodes = 200     # number of nodes
max_iter = 200   # maximum number of iterations
c = 1           # penalty parameter in ADMM
v = np.random.rand(n_nodes) * 10 + 10
x_opt = v.mean()
x0 = np.random.randn(n_nodes)
setting = {'penalty': c, 'max_iter':max_iter, 'objective':v, 'initial':x0}

# generate graph
graph_type = 'Star Graph'

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
sim.simulation_setting['penalty']  = 1
c_opt_gap, c_primal_residual, c_dual_residual = sim.run_least_squares()

# hybrid
sim.mode = 'hybrid'
sim.simulation_setting['penalty']  = 10
h_opt_gap, h_primal_residual, h_dual_residual = sim.run_least_squares()

# decentralized ADMM
sim.mode = 'decentralized'
sim.simulation_setting['penalty']  = 10
d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()


marker_at = range(0, max_iter, 10)
title_str = '{}, N={}'.format(graph_type, n_nodes)
plt.figure(1)
plt.semilogy(d_opt_gap, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.semilogy(c_opt_gap, '-s', lw=2, label='centralized', markevery=marker_at)
plt.semilogy(h_opt_gap, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.ylabel('Optimality gap $||x - x^\star||^2$')
plt.xlabel('Iterations')
plt.title(title_str)
plt.legend()


plt.figure(2)
plt.plot(d_primal_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.plot(c_primal_residual, '-s', lw=2, label='centralized', markevery=marker_at)
plt.plot(h_primal_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.title(title_str)
plt.xlabel('Iterations')
plt.ylabel('Primal residual')
plt.legend()

plt.figure(3)
plt.plot(d_dual_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.plot(c_dual_residual, '-s', lw=2, label='centralized', markevery=marker_at)
plt.plot(h_dual_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.title(title_str)
plt.xlabel('Iterations')
plt.ylabel('Dual residual')
plt.legend()


plt.show()
