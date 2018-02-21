import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


a = np.diag(np.ones(19,), k=1)
a = a + a.T
la = np.diag(np.sum(a, axis=0)) - a
[v, d] = linalg.eig(la)
# print(a)