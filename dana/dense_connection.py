#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
'''
DenseConnection

'''
import numpy as np
from functions import extract
from functions import extract, convolution_matrix
from connection import Connection, ConnectionError


class DenseConnection(Connection):
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

        if weights.shape == (self.target.size, self.source.size):
            self._weights = weights
            self._mask = 1-np.isnan(self._weights).astype(np.int32)
            if self._mask.all():
                self._mask = 1
            return

        if len(weights.shape) != len(weights.shape):
            raise ConnectionError, \
                'Weights matrix shape is wrong relative to source and target'

        K = convolution_matrix(self.source, self.target, weights, self._toric)
        nz_rows = K.row
        nz_cols = K.col
        self._weights = K.todense()
        self._mask = np.zeros(K.shape)
        self._mask[nz_rows, nz_cols] = 1
        if self._mask.all(): self._mask = 1
        self._weights = np.array(K.todense())


    def output(self):
        ''' '''
        R = np.dot(self._weights, self._actual_source.ravel()) 
        return R.reshape(self._target.shape)


    def evaluate(self, dt=0.01):
        ''' Update weights relative to connection equation '''
        if not self._equation:
            return
        self._equation.evaluate(self._weights, dt, **self._kwargs)
        if self._mask is not 1:
            self._weights *= self._mask


    def __getitem__(self, key):
        ''' '''
        src = self.source
        dst = self.target
        to_flat_index = np.ones(len(dst.shape), dtype=int)
        to_flat_index[:-1] = dst.shape[:-1]
        index = (key*to_flat_index).sum()
        weights = np.array(self._weights[index]).ravel()
        mask = self._mask
        if mask is not 1:
            nz = np.array(self._mask[index]).ravel().nonzero()
            masked_weights = np.zeros(weights.size)*np.NaN
            masked_weights[nz] = weights[nz]
            return masked_weights.reshape(self.source.shape)
        else:
            return weights.reshape(self.source.shape)

       
