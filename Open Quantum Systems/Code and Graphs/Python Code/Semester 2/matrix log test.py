

import numpy as np
a = np.array([[1,0],[0,0]])
b= np.array([[0,0],[0,1]])

from scipy.linalg import logm, expm
print(logm(a))
print(logm(b))

print(a.dot(logm(a)))
print(b.dot(logm(b)))

print(expm(logm(a)))