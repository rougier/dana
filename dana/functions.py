#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
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
def gaussian(shape=(25,25), width=0.5, center=0.0):
    ''' Generate a gaussian of the form g(x) = height*exp(-(x-center)**2/width**2).

    **Parameters**

        shape: tuple of integers
           Shape of the output array

        width: float or tuple of float
           Width of gaussian

        center: float or tuple of float
           Center of gaussian

    **Returns**

       a numpy array of specified shape containing a gaussian
    '''
    if type(shape) in [float,int]:
        shape = (shape,)
    if type(width) in [float,int]:
        width = (width,)*len(shape)
    if type(center) in [float,int]:
        center = (center,)*len(shape)
    grid=[]
    for size in shape:
        grid.append (slice(0,size))
    C = numpy.mgrid[tuple(grid)]
    R = numpy.zeros(shape)
    for i,size in enumerate(shape):
        R += (((C[i]/float(size-1))*2 - 1 - center[i])/width[i])**2
    return numpy.exp(-R/2)

def empty(shape, dtype=float, order='C'):
    '''
    Return a new group of given shape and type, without initialising entries.

    **Parameters**

    shape : {tuple of ints, int}
        Shape of the new group, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the group, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    
    **Returns**

    out : group
        Group with the given shape, dtype, and order.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.zeros_like` : Return a group of zeros with shape and type of input.
    * :meth:`dana.ones_like` : Return a group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return a empty group with shape and type of input.

    **Notes**

    `empty`, unlike `zeros`, does not set the group values to zero, and may
    therefore be marginally faster.  On the other hand, it requires the user
    to manually set all the values in the group, and should be used with
    caution.

    **Examples**

    >>> Group.empty((2,2))
    Group([[6.94248367807e-310, 1.34841898023e-316],
           [1.34841977073e-316, 0.0]], 
          dtype=[('f0', '<f8')])
    '''
    return Group(shape=shape, dtype=dtype, order=order, fill=None)

def zeros(shape, dtype=float, order='C'):
    '''
    Return a new group of given shape and type, filled with zeros.

    **Parameters**

    shape : {tuple of ints, int}
        Shape of the new group, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the group, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    
    **Returns**

    out : group
        Group with the given shape, dtype, and order, filled with zeros.

    **See also**

    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.zeros_like` : Return an group of zeros with shape and type of input.
    * :meth:`dana.ones_like` : Return an group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.

    **Examples**

    >>> dana.zeros((2,2))
    Group([[0.0, 0.0],
           [0.0, 0.0]], 
          dtype=[('f0', '<f8')])
    >>> dana.zeros((2,2), dtype=int)
    Group([[0, 0],
           [0, 0]], 
          dtype=[('f0', '<f8')])
    '''
    return Group(shape=shape, dtype=dtype, order=order, fill=0)

def ones(shape, dtype=float, order='C'):
    '''
    Return a new group of given shape and type, filled with ones.

    **Parameters**

    shape : {tuple of ints, int}
        Shape of the new group, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the group, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    
    **Returns**

    out : group
        Group with the given shape, dtype, and order, filled with ones.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.zeros_like` : Return an group of zeros with shape and type of input.
    * :meth:`dana.ones_like` : Return an group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.

    **Examples**

    >>> dana.ones((2,2))
    Group([[1.0, 1.0],
           [1.0, 1.0]], 
          dtype=[('f0', '<f8')])
    >>> dana.ones((2,2), dtype=int)
    Group([[1, 1],
           [1, 1]], 
          dtype=[('f0', '<f8')])
    '''
    return Group(shape=shape, dtype=dtype, order=order, fill=1)

def empty_like(other):
    ''' 
    Create a new group with the same shape and type as another.

    **Parameters**

    other : array_like
        The shape and data-type of `other` defines the parameters of the
        returned group.

    **Returns**

    out : group
        Unintialized group with same shape and type as `other`.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.ones_like` : Return a group of ones with shape and type of input.
    * :meth:`dana.zeros_like` : Return a group of zeros with shape and type of input.

    **Examples**

    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
    '''
    return Group(shape=other.shape, dtype=other.dtype, fill=None)

def zeros_like(other):
    ''' 
    Create a new group of zeros with the same shape and type as another.

    **Parameters**

    other : array_like
        The shape and data-type of `other` defines the parameters of the
        returned group.

    **Returns**

    out : group
        Group of zeros with same shape and type as `other`.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.empty_like` : Return an uninitialized group shape and type of input.
    * :meth:`dana.ones_like` : Return a group of ones with shape and type of input.

    **Examples**

    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
    '''
    return Group(shape=other.shape, dtype=other.dtype, fill=0)

def ones_like(other):
    '''
    Returns a group of ones with the same shape and type as a given array.

    **Parameters**

    other : group_like
        The shape and data-type of other defines the parameters of the
        returned group.

    **Returns**

    out : group
        Group of ones with same shape and type as other.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.
    * :meth:`dana.zeros_like` : Return a group of zeros with shape and type of input.

    **Examples**

    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> zeros_like(x)
    group([[0, 0, 0],
           [0, 0, 0]])
    '''
    return Group(shape=other, dtype=other.dtype, fill=1)
