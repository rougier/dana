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
import inspect
import scipy
import numpy as np
from functions import extract, convolve1d, convolve2d
from connection import Connection, ConnectionError


class SharedConnection(Connection):
    ''' '''

    def __init__(self, source=None, target=None, weights=None, toric=False):
        ''' '''

        Connection.__init__(self, source, target, toric)
        self._src_rows = None
        self._src_cols = None
        self.setup_weights(weights)
        self.setup_equation(None)


    def setup_weights(self, weights):
        ''' Setup weights '''

        # 1d convolution case
        # -------------------
        if len(self.source.shape) == len(self.target.shape) == 1:
            if len(weights.shape) != 1:
                raise ConnectionError, \
                 '''Shared connection requested but weights matrix shape does not match.'''
            if self.source.shape != self.target.shape:
                rows = np.rint((np.linspace(0,1,self.target.shape[0])\
                                 *(self.source.shape[0]-1))).astype(int)
                self._src_rows = rows
            self._weights = np.nan_to_num(weights)

        # 2d convolution case
        # -------------------
        elif len(self.source.shape) == len(self.target.shape) == 2:
            if len(weights.shape) != 2:
                raise ConnectionError, \
                    '''Shared connection requested but weights matrix shape does not match.'''
            if self.source.shape != self.target.shape:
                rows = np.rint((np.linspace(0,1,self.target.shape[0])\
                                 *(self.source.shape[0]-1))).astype(int)
                cols = np.rint((np.linspace(0,1,self.target.shape[1])\
                                 *(self.source.shape[1]-1))).astype(int)
                self._src_rows = rows.reshape((len(rows),1))
                self._src_cols = cols.reshape((1,len(cols)))

            self._weights = np.nan_to_num(weights)
            dtype = weights.dtype
            self._USV = scipy.linalg.svd(np.nan_to_num(weights))
            U,S,V = self._USV
            self._USV = U.astype(dtype), S.astype(dtype), V.astype(dtype)

        # Higher dimensional case
        # ------------------------
        else:
            raise ConnectionError, \
                '''Shared connection requested but dimensions are too high (> 2).'''


    def output(self):
        ''' '''
        if len(self._source.shape) == 1:
            if self._src_rows is not None:
                source = self._actual_source[self._src_rows]
            else:
                source = self._actual_source
            R = convolve1d(source, self._weights, self._toric)
        else:
            if self._src_rows is not None and self._src_cols is not None:
                source = self._actual_source[self._src_rows, self._src_cols]
                source = source.reshape(self.target.shape)
            else:
                source = self._actual_source
            R = convolve2d(source, self._weights, self._USV, self._toric)
        return R.reshape(self._target.shape)


    def __getitem__(self, key):
        ''' '''

        key = np.array(key) % self.target.shape
        s = np.array(list(self.source.shape)).astype(float)/np.array(list(self.target.shape))
        c = (key*s).astype(int).flatten()
        Ks = np.array(list(self.weights.shape), dtype=int)//2
        Ss = np.array(list(self.source.shape), dtype=int)//2
        return extract(self.weights, self.source.shape, Ks+Ss-c, np.NaN)


# ---------------------------------------------------------------- __main__ ---
if __name__ == '__main__':
    from random import choice

    # 1. SharedConnection example using regular arrays
    # -------------------------------------------------------------------------
    source = np.ones((3,3))
    target = np.ones((3,3))

    # 1.1 Without learning, full specification of the weights
    # -------------------------------------------------------
    weights = np.ones((target.size,source.size))
    C = SharedConnection(source, target, weights)
    C.propagate()
    print target
    print C[0,0]

    # 1.2 Without learning, partial specification of the weights
    # ----------------------------------------------------------
    weights = np.ones((3,3))
    C = SharedConnection(source, target, weights)
    C.propagate()
    print target
    print C[0,0]

    # 2. SharedConnection example using record arrays
    # -------------------------------------------------------------------------
    source = np.ones((3,3), dtype=[('V','f8'), ('mask','f8')])
    target = np.ones((3,3), dtype=[('V','f8'), ('mask','f8')])
    weights = np.ones((3,3))

    # 2.1 Without learning, without mask
    # ----------------------------------
    C = SharedConnection(source, target, weights)
    C.propagate()
    print target['V']
    print C[0,0]

    # 2.2 Without learning, with mask in source
    # -----------------------------------------
    source['mask'][0,0] = 0
    C = SharedConnection(source, target, weights)
    C.propagate()
    print target['V']
    print C[0,0]

    # 2.3 Without learning, with mask in weights
    # ------------------------------------------
    source['mask'] = 1
    weights[1,1] = np.NaN
    C = SharedConnection(source, target, weights)
    C.propagate()
    print target['V']
    print C[0,0]
