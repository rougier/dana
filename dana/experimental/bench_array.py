import numpy as np
import dana
from random import sample
from scipy import random, sparse
import matplotlib.pyplot as plt


def sparse_random_matrix(dst_size, src_size, s0, s1, p=0.1, value=1.):
    W = sparse.lil_matrix((dst_size, src_size))
    for i in xrange(dst_size):
        k = random.binomial(s1-s0,p,1)[0]
        W.rows[i] = sample(xrange(s0,s1),k)
        W.rows[i].sort()
        W.data[i] = [value]*k
    return W


n = 10
G = dana.zeros((n,))
W = (sparse_random_matrix(len(G), len(G), int(0.0*n), int(0.8*n), 0.02, 1) + 
     sparse_random_matrix(len(G), len(G), int(0.8*n), int(1.0*n), 0.02, 1))


print W.todense()
print
print G.V.reshape((n,1))

print W.get_rows(1)

V = sparse.csr_matrix(G.V.reshape((n,1)))

#print W.shape
#print V.shape
#print timeit.Timer('a=data.ages; a += 1','from __main__ import data')
#timer = timeit.Timer('W*G.V','from __main__ import W, G')
#print timer.timeit(number=100)

#print timeit.Timer('W*G.V','from __main__ import W, G').timeit()
