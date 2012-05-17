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
DenseConnection

"""
import numpy as np
from functions import extract
from functions import extract, convolution_matrix
from connection import Connection, ConnectionError


class DenseConnection(Connection):
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

        if weights.shape == (self.target.size, self.source.size):
            self._weights = weights
            self._mask = 1-np.isnan(self._weights).astype(np.int32)
            if self._mask.all():
                self._mask = 1
            return

        if len(weights.shape) != len(weights.shape):
            raise ConnectionError, \
                'Weights matrix shape is wrong relative to source and target'

        # If we have a toric connection, weights cannot be greater than source
        # in any dimension
        if self._toric:
            s = np.array(self.source.shape)
            w = np.array(weights.shape)
            weights = extract(weights, np.minimum(s,w), w//2)

        K = convolution_matrix(self.source, self.target, weights, self._toric)
        nz_rows = K.row
        nz_cols = K.col
        self._weights = K.todense()
        self._mask = np.zeros(K.shape)
        self._mask[nz_rows, nz_cols] = 1
        if self._mask.all(): self._mask = 1
        self._weights = np.array(K.todense())


    def output(self):
        """ """

        R = np.dot(self._weights, self._actual_source.ravel()) 
        return R.reshape(self._target.shape)


    def evaluate(self, dt=0.01):
        """ Update weights relative to connection equation """
        if not self._equation:
            return
        Connection.evaluate(self,dt)
        if self._mask is not 1:
            self._weights *= self._mask


    def __getitem__(self, key):
        """ """
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

       
