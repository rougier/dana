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
SharedConnection

"""
import inspect
import scipy
import numpy as np
from functions import extract, convolve1d, convolve2d, best_fft_shape
from connection import Connection, ConnectionError
from numpy.fft import fft, ifft
from numpy.fft import fft2, ifft2
from numpy.fft import rfft, irfft
from numpy.fft import rfft2, irfft2
from numpy.fft import fftshift, ifftshift
#from scipy.fftpack import fft, ifft, fft2, ifft2
#from numpy import fftshift, ifftshift
#from scipy.fftpack import rfft, irfft, rfft2, irfft2


class SharedConnection(Connection):
    """ """

    def __init__(self, source=None, target=None, weights=None, toric=False, fft=True):
        """ """

        Connection.__init__(self, source, target, toric)
        self._src_rows = None
        self._src_cols = None
        self._fft = fft
        self.setup_weights(weights)
        self.setup_equation(None)


    def setup_weights(self, weights):
        """ Setup weights """

        # If we have a toric connection, kernel cannot be greater than source
        # in any dimension
        if self._toric:
            s = np.array(self.source.shape)
            w = np.array(weights.shape)
            weights = extract(weights, np.minimum(s,w), w//2)

        # 1d convolution case
        # -------------------
        if len(self.source.shape) == len(self.target.shape) == 1:
            if len(weights.shape) != 1:
                raise ConnectionError, \
                 '''Shared connection requested but weights matrix shape does not match.'''
            if self.source.shape != self.target.shape:
                rows = np.rint((np.linspace(0,1,self.target.shape[0])
                                *(self.source.shape[0]-1))).astype(int)
                self._src_rows = rows

            if self._fft:
                src_shape = np.array(self.source.shape)
                wgt_shape = np.array(weights.shape)
                K = np.nan_to_num(weights)[::-1]
                if self._toric:
                    K_ = extract(K, src_shape, wgt_shape//2)
                    self._fft_weights = rfft(ifftshift(K_))
                else:
                    size = src_shape+wgt_shape//2
                    shape = best_fft_shape(size)
                    self._fft_weights = rfft(K,shape[0])
                    i0 = wgt_shape[0]//2
                    i1 = i0+src_shape[0]
                    self._fft_indices = slice(i0,i1)
                    self._fft_shape = shape

                # m = self.source.shape[0]
                # p = weights.shape[0]
                # if self._toric:
                #     _weights = extract(weights[::-1], (m,), (np.floor(p/2.0),) )
                # else:
                #     self._src_holder = np.zeros(2*m+1)
                #     _weights = extract(weights[::-1], (2*m+1,), (np.floor(p/2.0),) )
                # self._fft_weights = fft(ifftshift(np.nan_to_num(_weights)))


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
                rows = np.rint((np.linspace(0,1,self.target.shape[0])
                                *(self.source.shape[0]-1))).astype(int)
                cols = np.rint((np.linspace(0,1,self.target.shape[1])
                                *(self.source.shape[1]-1))).astype(int)
                self._src_rows = rows.reshape((len(rows),1))
                self._src_cols = cols.reshape((1,len(cols)))

            if self._fft:
                src_shape = np.array(self.source.shape)
                wgt_shape = np.array(weights.shape)
                K = np.nan_to_num(weights)[::-1,::-1]
                if self._toric:
                    K_ = extract(K, src_shape, wgt_shape//2)
                    self._fft_weights = rfft2(ifftshift(K_))
                else:
                    size = src_shape+wgt_shape//2
                    shape = best_fft_shape(size)
                    self._fft_weights = rfft2(K,shape)
                    i0 = wgt_shape[0]//2
                    i1 = i0+src_shape[0]
                    j0 = wgt_shape[1]//2
                    j1 = j0+src_shape[1]
                    self._fft_indices = slice(i0,i1),slice(j0,j1)
                    self._fft_shape = shape

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
        """ """

        # One dimension
        if len(self._source.shape) == 1:
            source = self._actual_source
            # Use FFT convolution
            if self._fft:
                if not self._toric:
                    P = rfft(source,self._fft_shape[0])*self._fft_weights
                    R = irfft(P, self._fft_shape[0]).real
                    R = R[self._fft_indices]
                else:
                    P = rfft(source)*self._fft_weights
                    R = irfft(P,source.shape[0]).real

                # if self._toric:
                #     R  = ifft(fft(source)*self._fft_weights).real
                # else:
                #     n = source.shape[0]
                #     self._src_holder[n//2:n//2+n] = source
                #     R = ifft(fft(self._src_holder)*self._fft_weights)
                #     R = R.real[n//2:n//2+n]
            # Use regular convolution
            else:
                R = convolve1d(source, self._weights[::-1], self._toric)
            if self._src_rows is not None:
                R = R[self._src_rows]
            return R.reshape(self._target.shape)
        # Two dimensions
        else:
            source = self._actual_source
            # Use FFT convolution
            if self._fft:
                if not self._toric:
                    P = rfft2(source,self._fft_shape)*self._fft_weights
                    R = irfft2(P, self._fft_shape).real
                    R = R[self._fft_indices]
                else:
                    P = rfft2(source)*self._fft_weights
                    R = irfft2(P,source.shape).real

            # Use SVD convolution
            else:
                R = convolve2d(source, self._weights, self._USV, self._toric)
            if self._src_rows is not None and self._src_cols is not None:
                R = R[self._src_rows, self._src_cols]
        return R.reshape(self._target.shape)


    def __getitem__(self, key):
        """ """

        src = self.source
        dst = self.target
        kernel = self._weights
        src_shape = np.array(src.shape, dtype=float)
        dst_shape = np.array(dst.shape, dtype=float)
        kernel_shape = np.array(kernel.shape, dtype=float)
        dst_key = np.array(key, dtype=float)
        src_key = np.rint((dst_key/(dst_shape-1))*(src_shape-1)).astype(int)
        scale = 1 #dst_shape/src_shape
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
