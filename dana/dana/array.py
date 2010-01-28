#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE
import numpy as np
import scipy.sparse as sp
from scipy.sparse import isspmatrix
from scipy.sparse.sputils import isdense, isscalarlike, isintlike


class csr_array(sp.csr_matrix):
    ''' Sparse array with a fixed mask '''

    def __init__(self, *args, **kwargs):
        ''' Build array and create mask based on non zero values '''
        sp.csr_matrix.__init__(self, *args, **kwargs)
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

        if other.ndim == 1 or other.ndim == 2 and other.shape[0] == 1:
            if other.shape == (N,):
                operand(other[mask[1]])
            elif other.shape == (1,N):
                operand(other[0,mask[1]])
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


    def __idiv__(self, other): # self -= other
        return self.__binary_op__(other, '__idiv__')

    def __div__(self, other):  # self - other
        return self.copy().__idiv__(other)

    def __rdiv__(self, other): # other - self
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



class array(np.ndarray):
    ''' An array object represents a multidimensional, homogeneous array of
    fixed-size items. An associated data-type object describes the format of
    each element in the array (its byte-order, how many bytes it occupies in
    memory, whether it is an integer or a floating point number, etc.).

    Arrays should be constructed using array, zeros or empty (refer to the See
    Also section below). The parameters given here describe a low-level method
    for instantiating an array (ndarray(...)).

    For more information, refer to the numpy module and examine the the methods
    and attributes of an array.
    '''

    _shape = np.ndarray.shape
    _base = None

    def __new__(subtype, shape=(1,1), dtype=np.double, buffer=None,
                offset=None,strides=None, order=None, base=None):
        ''' Create an array.
        
        :Parameters:
            ``shape`` : tuple of ints
                Shape of created array.
            ``dtype`` : data type, optional
                Any object that can be interpreted a numpy data type.
            ``buffer`` : object exposing buffer interface, optional
                Used to fill the array with data.
            ``offset`` : int, optional
                Offset of array data in buffer.
            ``strides`` : tuple of ints, optional
                Strides of data in memory.
            ``order`` : {'C', 'F'}, optional
                Row-major or column-major order.
            ``base`` : group
                Base group of this array

        Returns
        -------
        out: array
           Array of given shape and type.
        '''
        obj = np.ndarray.__new__(subtype, shape=shape, dtype=dtype)
        obj._base = base
        return obj

    def _get_base(self):
        return self._base or self
    base = property(_get_base,
                     doc = '''Base group this array belongs to.''')

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self,key,value)
        base = self._base
        if (not base or not hasattr(base, '_values') 
            or 'mask' not in base._values.keys()):
            return
        if id(self) == id(base['mask']):
            for k in base._values.keys():
                if k != 'mask':
                    v = self._base._values[k]
                    v[...] = np.nan_to_num(v)
                    v += np.where(base.mask, 0, np.nan)
        else:
            self += np.where(base.mask, 0, np.nan)


    def _force_shape(self, shape):
        self._shape = shape
    def _get_shape(self):
        return self._shape
    def _set_shape(self, shape):
        if shape == self._shape:
            return
        if self.base == None:
            self._shape = shape
        else:
            raise AttributeError, \
               '''Cannot reshape a child array (''base'' is not None)'''
    shape = property(_get_shape, _set_shape,
                     doc = '''Tuple of array dimensions.\n
                              **Examples**

                              >>> x = dana.group((1,2))
                              >>> x.shape
                              (2,)
                              >>> y = dana.zeros((4,5,6))
                              >>> y.shape
                              (4, 5, 6)
                              >>> y.shape = (2, 5, 2, 3, 2)
                              >>> y.shape
                              (2, 5, 2, 3, 2)''')
