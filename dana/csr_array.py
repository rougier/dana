#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
from scipy.sparse import isspmatrix
from scipy.sparse.sputils import isdense, isscalarlike, isintlike


class csr_array(sp.csr_matrix):
    ''' Sparse array with a fixed mask '''

    def __init__(self, *args, **kwargs):
        ''' Build array and create mask based on non zero values '''
        sp.csr_matrix.__init__(self, *args, **kwargs)
        #self.mask = (1-np.isnan(self.data)).nonzero()
        self.mask = self.nonzero()


    def __binary_op__ (self, other, operand):
        ''' Generic binary op (+,-,/,*) implementation '''
        
        M,N = self.shape
        operand = getattr(self.data, operand)
        mask = self.mask
        if isscalarlike(other):
            operand(other)
            return self
        elif sp.issparse(other):
            if (other.shape != self.shape):
                raise ValueError, "inconsistent shapes"
            data = np.array(sp.lil_matrix(other)[mask].todense())
            operand(data.reshape(data.size))
            return self
        try:
            other.shape
        except AttributeError:
            other = np.asanyarray(other)
        other = np.asanyarray(other)

        if other.size == 1:
            operand(other[0])
            return self

        if other.ndim == 1 or other.ndim == 2 and other.shape[0] == 1:
            if other.shape == (N,):
                operand(other[mask[1]])
            elif other.shape == (1,N):
                operand(other[0,mask[1]])
#            elif len(self.data) == other.size:
#                operand(other.flatten())
            else:
                raise ValueError('dimension mismatch')
        elif other.ndim == 2:
            if other.shape == (M,1):
                operand(other[mask[0],0])
            elif other.shape == (M,N):
                operand(other[mask[0],mask[1]])
            else:
                raise ValueError('dimension mismatch')
        else:
            raise ValueError('could not interpret dimensions')
        return self


    def __iadd__(self, other): # self += other
        return self.__binary_op__(other, '__iadd__')

    def __add__(self, other):  # self + other
        return self.copy().__iadd__(other)

    def __radd__(self, other): # other + self
        return self.copy().__iadd__(other)


    def __isub__(self, other): # self -= other
        return self.__binary_op__(other, '__isub__')

    def __sub__(self, other):  # self - other
        return self.copy().__isub__(other)

    def __rsub__(self, other): # other - self
        return self.copy().__neg__().__iadd__(other)


    def __imul__(self, other): # self *= other
        return self.__binary_op__(other, '__imul__')

    def __mul__(self, other):  # self - other
        return self.copy().__imul__(other)

    def __rmul__(self, other): # other - self
        return self.copy().__imul__(other)


    def __idiv__(self, other): # self /= other
        return self.__binary_op__(other, '__idiv__')

    def __div__(self, other):  # self / other
        return self.copy().__idiv__(other)

    def __rdiv__(self, other): # other / self
        Z = self.copy()
        Z.data = 1.0/self.data
        return Z.__imul__(other)


    def __neg__(self):
        return self.copy().__mul__(-1)

    def cos(self):
        Z = self.copy()
        Z.data = np.cos(Z.data)
        return Z

    def sin(self):
        Z = self.copy()
        Z.data = np.sin(Z.data)
        return Z

    def exp(self):
        Z = self.copy()
        Z.data = np.exp(Z.data)
        return Z

    def sqrt(self):
        Z = self.copy()
        Z.data = np.sqrt(Z.data)
        return Z

    def abs(self):
        Z = self.copy()
        Z.data = np.abs(Z.data)
        return Z

    def sum(self, axis=None):
        '''Sum the matrix over the given axis.  If the axis is None, sum
        over both rows and columns, returning a scalar.
        '''

        Z = sp.csr_matrix(self)
        return Z.sum(axis)

        # M,N = self.shape
        # mask = self.mask
        # if axis == 0:
        #     # sum over columns
        #     return np.array(
        #         [np.where(mask[1]==i, self[mask[0],mask[1]],0).sum() for i in range(N)])
        # elif axis == 1:
        #     # sum over rows
        #     return np.array(
        #         [np.where(mask[0]==i,self[mask[0],mask[1]],0).sum() for i in range(M)])
        # elif axis is None:
        #     # sum over rows and columns
        #     return np.sum(self.data)
        # else:
        #     raise ValueError, "axis out of bounds"


def dot(A,B):
    ''' dot product AxB '''

    return sp.csr_matrix.__mul__(A,B)
    #._mul_sparse_matrix(B)).todense()
    #return (A._mul_sparse_matrix(B)).todense()


