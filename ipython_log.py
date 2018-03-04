# IPython log file

runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
get_ipython().run_line_magic('matplotlib', '')
n_markers = 20
marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)
line_style = ['-d', '--d', '-s', '--s']
# fig = plt.figure(1, figsize=(8, 6))
fig = plt.figure()
for data, style in zip(sim_data, line_style):
   plt.semilogy(data['opt_gap'], style, lw=2, label=data['legend'], markevery=marker_at)


plt.ylabel('Accuracy')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=1e-8)
# plt.grid()
plt.legend()

# f1.savefig(graph_type + '.pdf', bbox_inches='tight')
fig.tight_layout()
# fig.savefig(graph_type + '.pdf', bbox='tight', pad_inches=0)
plt.show()
n_markers = 20
marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)
line_style = ['-d', '--d', '-s', '--s']

# accuracy vs iteration
# fig = plt.figure(1, figsize=(8, 6))
fig = plt.figure()
for data, style in zip(sim_data, line_style):
   plt.semilogy(data['opt_gap'], style, lw=2, label=data['legend'], markevery=marker_at)


plt.ylabel('Accuracy')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=1e-8)
plt.legend()


fig.tight_layout()
plt.show()
runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
get_ipython().run_line_magic('reset', '')
runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
n_markers = 20
marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)

# accuracy vs iteration
# fig = plt.figure(1, figsize=(8, 6))
fig = plt.figure()
for data, style in zip(sim_data, line_style):
   plt.semilogy(data['opt_gap'], style, lw=2, label=data['legend'], markevery=marker_at)


plt.ylabel('Accuracy')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=1e-8)
plt.legend()


fig.tight_layout()
plt.show()
n_nodes = 50     # number of nodes
d = 3            # dimension of variable at each node

# 2. function
# objective value
v = np.random.rand(n_nodes, d)
# optimal value
x_opt = v.mean()

#np.random.seed(10)
#degree = nx.degree(g)
#print('degree = ', degree)
#plt.figure(100)
#nx.draw_networkx(g)

graphs = [nx.path_graph(n_nodes), nx.complete_graph(n_nodes),
          nx.star_graph(n_nodes-1)]
graph_name = ['line graph', 'complete graph', 'star graph']
line_style = ['-d', '--d', '-s', '--s', '-o', '--o']

# simulation
max_iter = 500
#title_str = '{}, Nodes: {}, Edges: {}'.format(graph_type, n_nodes, 
#             g.number_of_edges())


# start simulation
setting = {'penalty': 1, 'max_iter': max_iter, 'objective': v, 
           'initial': 0 * np.random.randn(n_nodes, d)}
G = graphs[0]
sim = Simulator(G, simulation_setting=setting)
best_penalty = (1, 8, 8)
all_mode = ['decentralized', 'hybrid']
n_markers = 20
marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)

# accuracy vs iteration
 fig = plt.figure(1, figsize=(8, 6))
#fig = plt.figure()
for data, style in zip(sim_data, line_style):
   plt.semilogy(data['opt_gap'], style, lw=2, label=data['legend'], markevery=marker_at)


plt.ylabel('Accuracy')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=1e-8)
plt.legend()


fig.tight_layout()
plt.show()
runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
n_markers = 20
marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)

# accuracy vs iteration
fig = plt.figure(1, figsize=(8, 6))
#fig = plt.figure()
for data, style in zip(sim_data, line_style):
   plt.semilogy(data['opt_gap'], style, lw=2, label=data['legend'], markevery=marker_at)


plt.ylabel('Accuracy')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=1e-8)
plt.legend()


fig.tight_layout()
plt.show()
sim_data = []
for G, name in zip(graphs, graph_name):
   sim = Simulator(G, simulation_setting=setting)
   for mode in all_mode:
      data = {}
      sim.mode = mode
      sim.simulation_setting['penalty'] = best_penalty[1]
      opt_gap, primal_residual, dual_residual = sim.run_least_squares()
      data['legend'] = name + ' ' + sim.mode
      data['opt_gap'] = opt_gap
      data['primal_residual'] = primal_residual
      data['dual_residual'] = dual_residual
      sim_data.append(data)
   
   # decentralized ADMM

#   sim.mode = 'decentralized'
#   sim.simulation_setting['penalty'] = best_penalty[2]
#   d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()
get_ipython().run_line_magic('reset', '')
from Simulator import Simulator
#from Admm_simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
n_nodes = 50     # number of nodes
d = 3            # dimension of variable at each node

# 2. function
# objective value
v = np.random.rand(n_nodes, d)
# optimal value
x_opt = v.mean()

#np.random.seed(10)
#degree = nx.degree(g)
#print('degree = ', degree)
#plt.figure(100)
#nx.draw_networkx(g)

graphs = [nx.path_graph(n_nodes), nx.complete_graph(n_nodes),
          nx.star_graph(n_nodes-1)]
graph_name = ['line graph', 'complete graph', 'star graph']
line_style = ['-d', '--d', '-s', '--s', '-o', '--o']

# simulation
max_iter = 500
#title_str = '{}, Nodes: {}, Edges: {}'.format(graph_type, n_nodes, 
#             g.number_of_edges())


# start simulation
setting = {'penalty': 1, 'max_iter': max_iter, 'objective': v, 
           'initial': 0 * np.random.randn(n_nodes, d)}
G = graphs[0]
sim = Simulator(G, simulation_setting=setting)
best_penalty = (1, 8, 8)
all_mode = ['decentralized', 'hybrid']
sim_data = []
for G, name in zip(graphs, graph_name):
   sim = Simulator(G, simulation_setting=setting)
   for mode in all_mode:
      data = {}
      sim.mode = mode
      sim.simulation_setting['penalty'] = best_penalty[1]
      opt_gap, primal_residual, dual_residual = sim.run_least_squares()
      data['legend'] = name + ' ' + sim.mode
      data['opt_gap'] = opt_gap
      data['primal_residual'] = primal_residual
      data['dual_residual'] = dual_residual
      sim_data.append(data)
   
   # decentralized ADMM

#   sim.mode = 'decentralized'
#   sim.simulation_setting['penalty'] = best_penalty[2]
#   d_opt_gap, d_primal_residual, d_dual_residual = sim.run_least_squares()
n_markers = 20
marker_at = range(0, setting['max_iter'], setting['max_iter'] // n_markers)

# accuracy vs iteration
fig = plt.figure(1, figsize=(8, 6))
#fig = plt.figure()
for data, style in zip(sim_data, line_style):
   plt.semilogy(data['opt_gap'], style, lw=2, label=data['legend'], markevery=marker_at)


plt.ylabel('Accuracy')
plt.xlabel('Iterations')
# plt.title(title_str)
plt.ylim(ymin=1e-8)
plt.legend()


fig.tight_layout()
plt.show()
get_ipython().run_line_magic('matplotlib', 'inline')
runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
runfile('C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM/Sim1_innet.py', wdir='C:/MyData/Dropbox/WorkFiles/PyProject/HCADMM')
get_ipython().run_line_magic('logon', '')
get_ipython().run_line_magic('logstart', '')
get_ipython().run_line_magic('logoff', '')
