


import time
import dana
import random
import numpy as np
from scipy import sparse as sp


def random_matrix(n,m,p,value=1.):
    W = sp.lil_matrix((n,m))
    for i in xrange(n):
        k=np.random.binomial(m,p,1)[0]
        W.rows[i]=random.sample(xrange(m),k)
        W.rows[i].sort()
        W.data[i]=[value]*k
    return W


n = 400

W = random_matrix(n,n,0.02)
W_sp   = sp.csr_matrix(W)
W_dana = dana.csr_array(W)

Z = np.random.random((n,1))
#A = np.random.random((n,1))

t = time.clock()
for i in range(5000):
    W_sp*Z
print time.clock()-t

t = time.clock()
for i in range(5000):
    W_dana*Z
print time.clock()-t

