import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from GraphGenerator import *
import numpy.linalg as LA

# 1. generate graph
# 2. generate parameters
# 3. generate hyper edges
# 4. run simulations

# generate hyper edges for line graph
n_nodes = 11
hyper_edges = [list(range(i, i + 3)) for i in range(0, n_nodes-1, 2)]
print(hyper_edges)
gg = GraphGenerator(n_nodes, hyper_edges=hyper_edges)
gg.line_graph()

# get A, B
A, B = gg.get_AB()
D_M = B.T.dot(B)
D_N = A.T.dot(A)
C = A.T.dot(B)

# generate v
x_star = np.random.randn(n_nodes) * .1
v = x_star + np.random.randn(n_nodes) * 0.01 + 2
x_opt = v.mean()

# simulation
n_iter = 200
x = np.random.randn(n_nodes)
z = LA.pinv(D_M).dot(C.T).dot(x)
alpha = np.zeros_like(x)
c = 1.
primal_gap = []

for i in range(n_iter):
    x = LA.pinv(1 + c * D_N).dot(v - alpha + c * C.dot(z))
    z = LA.inv(D_M).dot(C.T).dot(x)
    alpha += c * (D_N.dot(x) - C.dot(z))

    primal_gap.append(LA.norm(x - x_opt))

plt.semilogy(primal_gap)
plt.ylabel('$x - x^*$')
plt.show()