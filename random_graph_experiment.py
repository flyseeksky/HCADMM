from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt


# simulation parameters
n_nodes = 100     # number of nodes
max_iter = 200   # maximum number of iterations
c = 20           # penalty parameter in ADMM
v = np.random.rand(n_nodes) * 10 + 10
gg = Simulator(n_nodes, penalty=c, max_iter=max_iter, v=v)


x_opt = v.mean()

x0 = np.random.randn(n_nodes)
alpha0 = np.zeros_like(x0)

# centralized
c_hyper_edges = [list(range(n_nodes))]
gg.hyper_edge_list = c_hyper_edges
c_opt_gap, c_primal_residual, c_dual_residual = gg.run_least_squares(x0)

# hybrid
g = Simulator.erdos_renyi(.2)
h_hyper_edges = Simulator.auto_discover_hyper_edge(g)
gg.hyper_edge_list = h_hyper_edges
h_opt_gap, h_primal_residual, h_dual_residual = gg.run_least_squares(x0)

# decentralized ADMM
# TODO get A and B for decentralized method
gg.hyper_edge_list = d_hyper_edge
d_opt_gap, d_primal_residual, d_dual_residual = gg.run_least_squares(x0)

# plot
marker_at = range(0, max_iter, 10)
plt.figure(1)
plt.semilogy(d_opt_gap, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.semilogy(c_opt_gap, '-s', lw=2, label='centralized', markevery=marker_at)
plt.semilogy(h_opt_gap, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.ylabel('Optimality gap $||x - x^\star||^2$')
plt.xlabel('Iterations')
plt.title('Line graph with {:2d} nodes'.format(n_nodes))
plt.legend()


plt.figure(2)
plt.plot(d_primal_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.plot(c_primal_residual, '-s', lw=2, label='centralized', markevery=marker_at)
plt.plot(h_primal_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.title('Primal Residual')
plt.xlabel('Iterations')
plt.ylabel('Primal residual')
plt.legend()

plt.figure(3)
plt.plot(d_dual_residual, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.plot(c_dual_residual, '-s', lw=2, label='centralized', markevery=marker_at)
plt.plot(h_dual_residual, '-o', lw=2, label='hybrid', markevery=marker_at)
plt.title('Dual Residual')
plt.xlabel('Iterations')
plt.ylabel('Dual residual')
plt.legend()


plt.show()
