#!/usr/bin/env python
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
'''
SparseConnection
'''
import numpy as np
import scipy.sparse as sp
from csr_array import csr_array, dot
from functions import extract
from connection import Connection, ConnectionError


class SparseConnection(Connection):
    ''' '''

    def __init__(self, source=None, target=None, weights=None, equation = ''):
        ''' '''

        Connection.__init__(self, source, target)
        self.setup_weights(weights)
        self.setup_equation(equation)


    def setup_weights(self, weights):
        ''' Setup weights '''

        if type(weights) in [int,float]:
            weights = np.ones((1,)*len(self.source.shape))*weights

        # Is kernel already a sparse array ?
        if sp.issparse(weights):
            if weights.shape != (self.target.size, self.source.size):
                raise ConnectionError, \
                    'weights matrix shape is wrong relative to source and target'
            else:
                self._weights = csr_array(weights, dtype=weights.dtype)
        elif weights.shape != (self.target.size,self.source.size):
            if len(weights.shape) == len(self.source.shape):
                K = sp.lil_matrix((self.target.size, self.source.size),
                                  dtype=weights.dtype)
                I = []
                J = []
                V = []
                Ks = np.array(list(weights.shape), dtype=int)//2
                Ss = np.array(list(self.source.shape), dtype=int)//2
                for i in range(K.shape[0]):
                    index = np.array(list(np.unravel_index(i, self.target.shape)))
                    index = (index/np.array(self.target.shape, dtype=float) \
                                 * np.array(self.source.shape)).astype(int)
                    k = extract(weights, self.source.shape, Ks+Ss - index, np.NaN)
                    k = k.flatten()
                    #J_ = k.nonzero()[0].tolist()
                    J_ =(1-np.isnan(k)).nonzero()[0].tolist()
                    I_ = [i,]*len(J_)
                    V_ = k[J_].tolist()
                    I += I_
                    J += J_
                    V += V_
                K = sp.coo_matrix((V,(I,J)), shape=K.shape)
                self._weights = csr_array(K, dtype=weights.dtype)
            else:
                raise ConnectionError, \
                    'weights matrix shape is wrong relative to source and target'
        else:
            self._weights = csr_array(weights, dtype=weights.dtype)


    def output(self):
        ''' '''
        R = dot(self._weights, self._actual_source.flatten())
        return R.reshape(self._target.shape)

        # src = self._source.flatten()
        # names = src.dtype.names
        # if names == None or 'mask' not in names:
        #     R = dot(self._weights, src)
        # else:
        #     R = dot(self._weights, src[names[0]]*src['mask'])
        # return R.reshape(self._target.shape)


    def __getitem__(self, key):
        key = np.array(key) % self.target.shape
        s = np.ones((len(self.target.shape),))
        for i in range(0,len(self.target.shape)-1):
            s[i] = self.target.shape[i]*s[i+1]
        index = int((s*key).sum()) 
        nz = self.weights.mask[1][np.where(self.weights.mask[0] == index)]
        a = np.ones_like(nz)*index
        nz = (a,nz)
        k = np.ones((self.source.size,))*np.NaN
        k[nz[1]] = self.weights[nz]
        return k.reshape(self.source.shape)



# ---------------------------------------------------------------- __main__ ---
if __name__ == '__main__':
    from random import choice

    # # 1. SparseConnection example using regular arrays
    # # -------------------------------------------------------------------------
    # source = np.ones((3,3))
    # target = np.ones((3,3))

    # # 1.1 Without learning, full specification of the weights
    # # -------------------------------------------------------
    # weights = np.ones((target.size,source.size))
    # C = SparseConnection(source, target, weights)
    # C.propagate()
    # print target
    # print C[0,0]

    # # 1.2 Without learning, partial specification of the weights
    # # ----------------------------------------------------------
    # weights = np.ones((3,3))
    # C = SparseConnection(source, target, weights)
    # C.propagate()
    # print target
    # print C[0,0]

    # # 1.3 With learning
    # # -----------------
    # C = SparseConnection(source, target, weights, "dW/dt = pre*post")
    # C.propagate(), C.evaluate()
    # print target
    # print C[0,0]



    # # 2. SparseConnection example using record arrays
    # # -------------------------------------------------------------------------
    # source = np.ones((3,3), dtype=[('V','f8'), ('mask','f8')])
    # target = np.ones((3,3), dtype=[('V','f8'), ('mask','f8')])
    # weights = np.ones((3,3))

    # # 2.1 Without learning, without mask
    # # ----------------------------------
    # C = SparseConnection(source, target, weights)
    # C.propagate()
    # print target['V']
    # print C[0,0]

    # # 2.2 Without learning, with mask in source
    # # -----------------------------------------
    # source['mask'][0,0] = 0
    # C = SparseConnection(source, target, weights)
    # C.propagate()
    # print target['V']
    # print C[0,0]

    # # 2.3 Without learning, with mask in weights
    # # ------------------------------------------
    # source['mask'] = 1
    # weights[1,1] = np.NaN
    # C = SparseConnection(source, target, weights)
    # C.propagate()
    # print target['V']
    # print C[0,0]

    # # 2.4 With learning, without mask
    # # -------------------------------
    # weights = np.ones((1,1))
    # C = SparseConnection(source, target, weights, "dW/dt = pre.V*post.V")
    # C.propagate(), C.evaluate()
    # print target['V']
    # print C[0,0]


    # # 3. Oja rule
    # # -------------------------------------------------------------------------
    # source = np.ones((2,))
    # target = np.ones((1,))
    # weights = np.ones((1,2))
    # C = SparseConnection(source, target, weights, 'dW/dt = post*(pre-post*W)')

    # def sample(theta, mu1, std1, mu2, std2):
    #     ''' Random sample according to an elliptical  gaussian distribution'''
    #     u1 = np.random.random()
    #     u2 = np.random.random()
    #     T1 = np.sqrt(-2.0*np.log(u1))*np.cos(2.0*np.pi*u2)
    #     T2 = np.sqrt(-2.0*np.log(u1))*np.sin(2.0*np.pi*u2)
    #     x = mu1 + (std1*T1 * np.cos(theta) - std2*T2 * np.sin(theta))
    #     y = mu2 + (std1*T1 * np.sin(theta) + std2*T2 * np.cos(theta))	
    #     return x,y

    # theta = -135.0 * np.pi / 180.0
    # for i in xrange(10000):
    #     source[...] = sample(theta, 0.0, 1.0, 0.0, 0.5)
    #     C.propagate()
    #     C.evaluate(dt=0.001)

    # # Given that the distribution is elliptical, its principal component
    # # should be oriented along the main axis of the distribution.
    # # Therefore the weights should be +/- (cos(theta), sin(theta))
    # print "Learned weights : ", C.weights.todense()
    # print "It should have converged to +/- (%f,%f)" % (np.cos(theta), np.sin(theta))


    # # 4. BCM rule
    # # -------------------------------------------------------------------------
    # n = 10
    # tau = 1.0
    # tau_bar = tau * 0.1
    # source = np.ones((n,))
    # target = np.ones((n,), dtype=[('C','f8'), ('T','f8'),('F','f8')])
    # weights = np.random.random((n,n))
    # C = SparseConnection(source, target['F'], weights,
    #                'dW/dt = pre*post.C*(post.C-post.T)')
    # stims = np.identity(n)
    # for i in xrange(10000):
    #     source[:] = choice(stims).reshape(source.shape)
    #     C.propagate()
    #     target['C'] += (target['F']-target['C'])*tau
    #     target['T'] += (target['C']**2-target['T'])*tau_bar
    #     C.evaluate(dt=0.01)

    # for i in range(n):
    #     print 'Unit %d: ' % i, (C.weights.todense()[i] > 1e-3).astype(int)

       


    # 4. BCM rule
    # -------------------------------------------------------------------------
    from group import Group
    from random import choice
    n = 10
    stims = np.identity(n)
    source = Group((n,), ''' V = choice(stims)    : float ''')
    target = Group((n,), ''' dC/dt = (F-C)*1.0    : float
                             dT/dt = (C**2-T)*0.1 : float
                             ----------------------------
                             F                    : float ''')
    C = SparseConnection(source, target('F'), np.random.random((n,n)),
                        'dW/dt = pre.V*post.C*(post.C-post.T)*0.01')
    source.setup()
    target.setup()
    for i in xrange(10000):
        source.run(dt=1)
        target.run(dt=1)
    for i in range(n):
        print 'Unit %d: ' % i, (C.weights.todense()[i] > 1e-1).astype(int)

       
