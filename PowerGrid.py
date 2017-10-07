from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Read Power Grid network
# TODO find user HOME folder
path = '../../Datasets/power.gml'
g = nx.read_gml(path, label='id')

# simulation parameters
n_nodes = g.order()                     # number of nodes
max_iter = 200                          # maximum number of iterations
c = 20                                  # penalty parameter in ADMM
v = np.random.rand(n_nodes) * 10 + 10
x_opt = v.mean()
x0 = np.random.randn(n_nodes)
setting = {'penalty': c, 'max_iter':max_iter, 'objective':v, 'initial':x0}


# start simulation
sim = Simulator(g, simulation_setting=setting)

# centralized
sim.mode = 'centralized'
c_opt_gap, c_primal_residual, c_dual_residual = sim.run_least_squares()

# hybrid
sim.mode = 'hybrid'
h_opt_gap, h_primal_residual, h_dual_residual = sim.run_least_squares()

# decentralized ADMM
sim.mode = 'decentralized'
d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()

#plotting figures
marker_at = range(0, max_iter, 10)
title_str = 'US Power Grid Network, Nodes: {}, Edges: {}'.format(g.order(), g.number_of_edges())
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
