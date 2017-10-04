from GraphGenerator import GraphGenerator
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt


# simulation parameters
n_nodes = 50
n_iter = 200
c = 20
gg = GraphGenerator(n_nodes)

v = np.random.rand(n_nodes) * 10 + 5
x_opt = v.mean()

x0 = np.random.randn(n_nodes)
alpha0 = np.zeros_like(x0)

# centralized
hyper_edges = [list(range(n_nodes))]
gg.hyper_edges = hyper_edges
c_primal_gap = gg.run_least_squares(n_iter, x0, c, v)

# hybrid
index = list(range(n_nodes))
hyper_edges = [index[i:i + 3] for i in range(0, n_nodes-1, 2)]
gg.hyper_edges = hyper_edges
hybrid_primal_gap = gg.run_least_squares(n_iter, x0, c, v)

# decentralized ADMM
decentralized_hyper_edge = [[i, i + 1] for i in range(n_nodes-1)]
gg.hyper_edges = decentralized_hyper_edge
decentralized_primal_gap = gg.run_least_squares(n_iter, x0, c, v)

plt.semilogy(hybrid_primal_gap, lw=2, label='hybrid')
plt.semilogy(decentralized_primal_gap, lw=2, label='decentralized')
plt.semilogy(c_primal_gap, lw=2, label='centralized')
plt.ylabel('$||x - x^*||$')
plt.xlabel('Number of iterations')
plt.title('Line graph with {:2d} nodes'.format(n_nodes))
plt.legend()
plt.show()
