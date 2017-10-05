from GraphSimulator import Simulator
import numpy.linalg as LA
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
hyper_edges = [list(range(n_nodes))]
gg.hyper_edges = hyper_edges
c_primal_gap, c_primal_residual, c_dual_residual = gg.run_least_squares(max_iter, x0)

# hybrid
index = list(range(n_nodes))
hyper_edges = [index[i:i + 3] for i in range(0, n_nodes-1, 2)]
gg.hyper_edges = hyper_edges
hybrid_primal_gap, h_primal_residual, h_dual_residual = gg.run_least_squares(max_iter, x0)

# decentralized ADMM
decentralized_hyper_edge = [[i, i + 1] for i in range(n_nodes-1)]
gg.hyper_edges = decentralized_hyper_edge
decentralized_primal_gap, d_primal_residual, d_dual_residual = gg.run_least_squares(max_iter, x0)

# plot
marker_at = range(0, max_iter, 10)
plt.figure(1)
plt.semilogy(decentralized_primal_gap, '-d', lw=2, label='decentralized', markevery=marker_at)
plt.semilogy(c_primal_gap, '-^', lw=2, label='centralized', markevery=marker_at)
plt.semilogy(hybrid_primal_gap, '-o', lw=2, label='hybrid', markevery=marker_at)
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
