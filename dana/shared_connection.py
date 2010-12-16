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
            self._mask = np.ones(weights.shape)
            self._mask[np.isnan(weights).nonzero()] = 0
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
            self._mask = np.ones(weights.shape)
            self._mask[np.isnan(weights).nonzero()] = 0
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

        src = self.source
        dst = self.target
        kernel = self._weights
        src_shape = np.array(src.shape, dtype=float)
        dst_shape = np.array(dst.shape, dtype=float)
        kernel_shape = np.array(kernel.shape, dtype=float)
        dst_key = np.array(key, dtype=float)
        src_key = np.rint((dst_key/(dst_shape-1))*(src_shape-1)).astype(int)
        scale = dst_shape/src_shape

        Z = np.zeros(src.shape) * np.NaN
        for i in range(kernel.size):
            k_key = np.array(np.unravel_index(i, kernel.shape))
            if self._toric:
                key = (src_key + scale*k_key - kernel_shape//2).astype(int) % src_shape
            else:
                key = (src_key + scale*k_key - kernel_shape//2).astype(int)
            bad = False
            for k in range(len(key)):
                if key[k] < 0 or key[k] >= Z.shape[k]: bad = True
            if not bad:
                if self._mask[tuple(k_key.tolist())]:
                    Z[tuple(key.tolist())] = kernel[tuple(k_key.tolist())]
        return Z

        # I = np.ogrid[ [slice(0, src_shape[i] )
        #                for i in range(src_shape.size)] ]

        # if not self._toric:
        #     key = np.array(key) % self.target.shape
        #     s = np.array(list(self.source.shape)).astype(float)/np.array(list(self.target.shape))
        #     c = (key*s).astype(int).flatten()
        #     Ks = np.array(list(self.weights.shape), dtype=int)//2
        #     Ss = np.array(list(self.source.shape), dtype=int)//2
        #     W = self.weights.copy()
        #     W[self._mask] = np.NaN
        #     return extract(W, self.source.shape, Ks+Ss-c, np.NaN)
        # else:
        #     src_shape = np.array(list(self.source.shape))
        #     kernel_shape = np.array(list(self.weights.shape))
        #     shape = np.maximum(src_shape, kernel_shape)
        #     W = np.ones(shape)*np.NaN
        #     I = np.ogrid[ [slice(0, src_shape[i] )
        #                    for i in range(src_shape.size)] ]
        #     print I
        #     return W
            


