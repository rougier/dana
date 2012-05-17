#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
"""
SparseConnection
"""
import numpy as np
import scipy.sparse as sparse
from csr_array import csr_array, dot
from functions import extract, convolution_matrix
from connection import Connection, ConnectionError


class SparseConnection(Connection):
    """ """

    def __init__(self, source=None, target=None, weights=None, equation = '', toric=False):
        """ """

        Connection.__init__(self, source, target, toric)
        self.setup_weights(weights)
        self.setup_equation(equation)


    def setup_weights(self, weights):
        """ Setup weights """

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
                # If we have a toric connection, weights cannot be greater than source
                # in any dimension
                if self._toric:
                    s = np.array(self.source.shape)
                    w = np.array(weights.shape)
                    weights = extract(weights, np.minimum(s,w), w//2)
                weights = convolution_matrix(self.source, self.target, weights, self._toric)
            else:
                raise ConnectionError, \
                    'weights matrix shape is wrong relative to source and target'
        self._weights = csr_array(weights, dtype=dtype)


    def output(self):
        """ """
        R = dot(self._weights, self._actual_source.ravel()) 
        return R.reshape(self._target.shape)


    def __getitem__(self, key):
        """ """
        
        src = self.source
        dst = self.target
        to_flat_index = np.ones(len(dst.shape), dtype=int)
        to_flat_index[:-1] = dst.shape[:-1]
        index = (key*to_flat_index).sum()
        w = self._weights[index].tocoo()
        W = np.array([np.NaN,]*self.source.size)
        W[w.col] = w.data
        return W.reshape(self.source.shape)

