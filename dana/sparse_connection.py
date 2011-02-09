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
import scipy.sparse as sparse
from csr_array import csr_array, dot
from functions import extract, convolution_matrix
from connection import Connection, ConnectionError


class SparseConnection(Connection):
    ''' '''

    def __init__(self, source=None, target=None, weights=None, equation = '', toric=False):
        ''' '''

        Connection.__init__(self, source, target, toric)
        self.setup_weights(weights)
        self.setup_equation(equation)


    def setup_weights(self, weights):
        ''' Setup weights '''

        if type(weights) in [int,float]:
            weights = np.ones((1,)*len(self.source.shape))*weights
        dtype = weights.dtype

        # Is kernel already a sparse array ?
        if sparse.issparse(weights):
            if weights.shape != (self.target.size, self.source.size):
                raise ConnectionError, \
                    'weights matrix shape is wrong relative to source and target'
            else:
                W = weights.tocoo()
                data, row, col  = W.data,W.row,W.col
                i = (1 - np.isnan(data)).nonzero()
                data, row, col = data[i], row[i], col[i]
                data = np.where(data, data, np.NaN)
                weights = sparse.coo_matrix((data,(row,col)), shape=W.shape)
                weights.data = np.nan_to_num(data)
        # Else, we need to build it
        elif weights.shape != (self.target.size,self.source.size):
            if len(weights.shape) == len(self.source.shape):
                weights = convolution_matrix(self.source, self.target, weights, self._toric)
            else:
                raise ConnectionError, \
                    'weights matrix shape is wrong relative to source and target'
        self._weights = csr_array(weights, dtype=dtype)


    def output(self):
        ''' '''
        R = dot(self._weights, self._actual_source.ravel()) 
        return R.reshape(self._target.shape)


    def __getitem__(self, key):
        ''' '''
        
        src = self.source
        dst = self.target
        to_flat_index = np.ones(len(dst.shape), dtype=int)
        to_flat_index[:-1] = dst.shape[:-1]
        index = (key*to_flat_index).sum()
        w = self._weights[index].tocoo()
        W = np.array([np.NaN,]*self.source.size)
        W[w.col] = w.data
        return W.reshape(self.source.shape)

