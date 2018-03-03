# from Simulator import Simulator
from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Line graph with a large diameter needs more iterations to achieve certain accuracy
# simulation parameters
np.random.seed(1)
n_nodes = 10     # number of nodes
d = 3            # dimension of variable at each node
v = np.random.rand(n_nodes, d)
x_opt = v.mean()
np.random.seed(10)

# generate graph
graph_type = 'Line Graph'


if graph_type == 'Line Graph':
    best_penalty = (1, 6.75, 8.95)
    max_iter = 500
    g = nx.path_graph(n_nodes)
elif graph_type == 'Erdos Renyi':
    max_iter = 500
    best_penalty = (1, 6.75, 8.95)
    prob = .8
    g = Simulator.erdos_renyi(n_nodes, prob, seed=1000)
elif graph_type == 'Star Graph':
    max_iter = 50
    best_penalty = (1, 2, 2)
    g = nx.star_graph(n_nodes - 1)
elif graph_type == 'Cycle Graph':
    g = nx.cycle_graph(n_nodes)
    max_iter = 500
    best_penalty = (1, 6.75, 8.95)
else:
    raise Exception('Unsupported graph type')

degree = nx.degree(g)
print('degree = ', degree)
plt.figure(100)
nx.draw_networkx(g)

title_str = '{}, Nodes: {}, Edges: {}'.format(graph_type, n_nodes, 
             g.number_of_edges())


# start simulation
setting = {'penalty': 1, 'max_iter': max_iter, 'objective': v, 
           'initial': 0 * np.random.randn(n_nodes, d)}
sim = Simulator(g, simulation_setting=setting)

# centralized
# sim.mode = 'centralized'
# sim.simulation_setting['penalty'] = best_penalty[0]
# c_opt_gap, c_primal_residual, c_dual_residual = sim.run_least_squares()

# hybrid
sim.mode = 'hybrid'
sim.simulation_setting['penalty'] = best_penalty[1]
h_opt_gap, h_primal_residual, h_dual_residual = sim.run_least_squares()

# decentralized ADMM
sim.mode = 'decentralized'
sim.simulation_setting['penalty'] = best_penalty[2]
d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()


marker_at = range(0, setting['max_iter'], setting['max_iter'] // 20)
# fig = plt.figure(1, figsize=(8, 6))
fig = plt.figure(figsize=(8, 6))
plt.semilogy(d_opt_gap, '-d', lw=2, label='decentralized', markevery=marker_at)
# plt.semilogy(c_opt_gap, '--', lw=2)
plt.semilogy(h_opt_gap, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.ylabel('Relative accuracy $||x - x^\star||^2/||x^\star||^2$')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=1e-8)
# plt.grid()
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
# plt.grid()
# plt.legend()

# f1.savefig(graph_type + '.pdf', bbox_inches='tight')
fig.tight_layout()
# fig.savefig(graph_type + '.pdf', bbox='tight', pad_inches=0)
plt.show()
