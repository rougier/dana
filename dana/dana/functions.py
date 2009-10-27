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
''' Some useful functions '''
import numpy
import scipy.linalg



def convolve1d( Z, K ):
    """ Discrete, clamped, linear convolution of two one-dimensional sequences.

        The convolution operator is often seen in signal processing, where it
        models the effect of a linear time-invariant system on a signal [1]_.
        In probability theory, the sum of two independent random variables is
        distributed according to the convolution of their individual
        distributions.
    
        **Parameters**

            Z : (N,) array_like
                First one-dimensional input array (input).
            K : (M,) array_like
                Second one-dimensional input array (kernel).

        **Returns**

            out : array
                Discrete, clamped, linear convolution of `Z` and `K`.

        **Note**

            The discrete convolution operation is defined as

            .. math:: (f * g)[n] = \sum_{m = -\infty}^{\infty} f[m] g[n-m]

        **References**

        .. [1] Wikipedia, "Convolution",
                      http://en.wikipedia.org/wiki/Convolution.
    """

    R = numpy.convolve(Z, K, 'same')
    i0 = 0
    if R.shape[0] > Z.shape[0]:
        i0 = (R.shape[0]-Z.shape[0])/2 + 1 - Z.shape[0]%2
    i1 = i0+ Z.shape[0]
    return R[i0:i1]



def convolve2d(Z, K, USV = None):
    """ Discrete, clamped convolution of two two-dimensional arrays.

        The convolution operator is often seen in signal processing, where it
        models the effect of a linear time-invariant system on a signal [1]_.
        In probability theory, the sum of two independent random variables is
        distributed according to the convolution of their individual
        distributions. If the kernel K is separable, it is decomposed using a
        singular value decomposition [2]_ and the computing is optimized
        accordingly (when rank n is inferior to S.size/2)
    
        **Parameters**

            Z : (N1,N2) array_like
                First two-dimensional input array (input).
            K : (M1,M2) array_like
                Second two-dimensional input array (kernel).

        **Returns**

            out : ndarray
                Discrete, clamped, linear convolution of `Z` and `K`.

        **Note**

            The discrete convolution operation is defined as

            .. math:: (f * g)[n] = \sum_{m = -\infty}^{\infty} f[m] g[n-m]

        **References**

        .. [1] Wikipedia, "Convolution",
                  http://en.wikipedia.org/wiki/Convolution.
        .. [2] Wikipedia, "Singular Value Decomposition",
                  http://en.wikipedia.org/wiki/Singular_value_decomposition."
    """

    if USV is None:
        U,S,V = scipy.linalg.svd(K)
        U,S,V = U.astype(K.dtype), S.astype(K.dtype), V.astype(K.dtype)
    else:
        U,S,V = USV
    n = (S > 1e-9).sum()
#    n = (S > 0).sum()
    R = numpy.zeros( Z.shape )
    for k in range(n):
        Zt = Z.copy() * S[k]
        for i in range(Zt.shape[0]):
            Zt[i,:] = convolve1d(Zt[i,:], V[k,::-1])
        for i in range(Zt.shape[1]):
            Zt[:,i] = convolve1d(Zt[:,i], U[::-1,k])
        R += Zt
    return R



# def extract(Z, shape, p, fill=0):
#     """ Extract a sub-array from Z with given shape and centered on p. If there
#         is not enough data to fill output, it is padded with fill value.
    
#         **Parameters**

#             `Z` : array_like
#                Input array.

#            `shape` : tuple
#                Shape of the output array

#            `p` : tuple
#                Position within Z

#            `fill` : scalar
#                Fill value

#         **Returns**

#             `out` : array_like
#                 Z slice with given shape and center
#     """
#     R = numpy.ones(shape)*fill
# #    c = [i-int(d/2)-(1-d%2) for i,d in zip(p, Z.shape)]
# #    c = [i-int(d/2)-(d%2) for i,d in zip(p, Z.shape)]
#     c = [i-int(d//2) for i,d in zip(p, Z.shape)]
#     r = [slice(max(0,p),  min(p+Zs,Rs)       ) for p,Zs,Rs in zip(c,Z.shape,R.shape)]
#     z = [slice(-min(0,p), Zs+min(0,Rs-(p+Zs))) for p,Zs,Rs in zip(c,Z.shape,R.shape)]
#     R[r] = Z[z]
#     return R



def extract(Z, shape, position, fill=0):
    """ Extract a sub-array from Z using given shape and centered on position.
        If some part of the sub-array is out of Z bounds, result will be padded
        with fill value.

        **Parameters**
            `Z` : array_like
               Input array.

           `shape` : tuple
               Shape of the output array

           `position` : tuple
               Position within Z

           `fill` : scalar
               Fill value

        **Returns**
            `out` : array_like
                Z slice with given shape and center

        **Examples**

        >>> Z = numpy.arange(0,16).reshape((4,4))
        >>> extract(Z, shape=(3,3), position=(0,0))
        [[ NaN  NaN  NaN]
         [ NaN   0.   1.]
         [ NaN   4.   5.]]

        Schema:

            +-----------+
            | 0   0   0 | = extract (Z, shape=(3,3), position=(0,0))
            |   +---------------+
            | 0 | 0   1 | 2   3 | = Z
            |   |       |       |
            | 0 | 4   5 | 6   7 |
            +---|-------+       |
                | 8   9  10  11 |
                |               |
                | 12 13  14  15 |
                +---------------+

        >>> Z = numpy.arange(0,16).reshape((4,4))
        >>> extract(Z, shape=(3,3), position=(3,3))
        [[ 10.  11.  NaN]
         [ 14.  15.  NaN]
         [ NaN  NaN  NaN]]

        Schema:

            +---------------+
            | 0   1   2   3 | = Z
            |               |
            | 4   5   6   7 |
            |       +-----------+
            | 8   9 |10  11 | 0 | = extract (Z, shape=(3,3), position=(3,3))
            |       |       |   |
            | 12 13 |14  15 | 0 |
            +---------------+   |
                    | 0   0   0 |
                    +-----------+
    """
#    assert(len(position) == len(Z.shape))
#    if len(shape) < len(Z.shape):
#        shape = shape + Z.shape[len(Z.shape)-len(shape):]

    R = numpy.ones(shape, dtype=Z.dtype)*fill
    P  = numpy.array(list(position)).astype(int)
    Rs = numpy.array(list(R.shape)).astype(int)
    Zs = numpy.array(list(Z.shape)).astype(int)

    R_start = numpy.zeros((len(shape),)).astype(int)
    R_stop  = numpy.array(list(shape)).astype(int)
    Z_start = (P-Rs//2)
    Z_stop  = (P+Rs//2)+Rs%2

    R_start = (R_start - numpy.minimum(Z_start,0)).tolist()
    Z_start = (numpy.maximum(Z_start,0)).tolist()
    #R_stop = (R_stop - numpy.maximum(Z_stop-Zs,0)).tolist()
    R_stop = numpy.maximum(R_start, (R_stop - numpy.maximum(Z_stop-Zs,0))).tolist()
    Z_stop = (numpy.minimum(Z_stop,Zs)).tolist()

    r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
    z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]

    R[r] = Z[z]

    return R
