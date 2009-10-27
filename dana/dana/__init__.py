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
'''
DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project

DANA is a python library based on numpy that support distributed, asynchronous,
numerical and adaptive computation which is closely related to both the notion
of artificial neural networks and cellular automaton. From a conceptual point of
view, the computational paradigm supporting the library is grounded on the
notion of a group that is a matrix of several arbitrary named values that can
vary along time under the influence of other groups and learning.

Contact:  CORTEX Project - INRIA
          INRIA Lorraine, 
          Campus Scientifique, BP 239
          54506 VANDOEUVRE-LES-NANCY CEDEX 
          FRANCE
'''
__version__ = '1.0'
__author__  = 'Nicolas Rougier'

import numpy
from group import group
from link import link

from tests import test
from functions import convolve1d, convolve2d, extract
from array import csr_array


def empty(shape, dtype=numpy.double, keys=['V'], mask=True, name=''):
    '''
    Return a new group of given shape and type, without initialising entries.

    **Parameters**
        shape : (m,n) tuple
            Shape of the empty group.
        dtype : data-type, optional
            Desired data-type.
        keys : list of strings, optional
            Names of the different keys
        mask : (m,n) array | bool, optional
            Group mask
        name : string, optional
            Group name

    **Returns**
        out : group
            Group of zeros with the given shape and dtype.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.zeros_like` : Return an group of zeros with shape and type of input.
    * :meth:`dana.ones_like` : Return an group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.


    **Notes**

        `empty`, unlike `zeros`, does not set the group values to zero, and may
        therefore be marginally faster.  On the other hand, it requires the user
        to manually set all the values in the group, and should be used with
        caution.

    **Examples**

    >>> dana.empty((2,2))
    group([[(7.5097298333540797e-316, True), (-6110.97607421875, True)],
           [(8.9749099642774968e-318, True), (2.1729236904796582e-310, True)]],
          dtype=[('V', '<f8'), ('mask', '|b1')])
    >>> dana.empty((2,2), dtype=int)
    group([[(-1073741821, True), (46148198, True)],
           [(276824155, True), (1679830018, True)]],
          dtype=[('V', '<i4'), ('mask', '|b1')])
    '''
    return group(shape=shape,dtype=dtype,keys=keys,
                 mask=mask,name=name,fill=None)


def zeros(shape, dtype=numpy.double, keys=['V'], mask=True, name=''):
    '''
    Return a new group of given shape and type, filled with zeros.

    **Parameters**

        shape : (m,n) tuple
            Shape of the empty group.
        dtype : data-type, optional
            Desired data-type.
        keys : list of strings, optional
            Names of the different keys
        mask : (m,n) array | bool, optional
            Group mask
        name : string, optional
            Group name

    **Returns**

        out : group
            Group of zeros with the given shape and dtype.

    **See also**

    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.zeros_like` : Return an group of zeros with shape and type of input.
    * :meth:`dana.ones_like` : Return an group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.

    **Examples**

    >>> dana.zeros((2,2))
    group([[(0.0, True), (0.0, True)],
           [(0.0, True), (0.0, True)]],
          dtype=[('V', '<f8'), ('mask', '|b1')])
    >>> dana.zeros((2,2), dtype=int)
    group([[(0, True), (0, True)],
           [(0, True), (0, True)]],
          dtype=[('V', '<f8'), ('mask', '|b1')])
    '''
    return group(shape=shape,dtype=dtype,keys=keys,
                 mask=mask,name=name,fill=0)


def ones(shape, dtype=numpy.double, keys=['V'], mask=True, name=''):
    '''
    Return a new group of given shape and type, filled with ones.

    **Parameters**

        shape : (m,n) tuple
            Shape of the empty group.
        dtype : data-type, optional
            Desired data-type.
        keys : list of strings, optional
            Names of the different keys
        mask : (m,n) array | bool, optional
            Group mask
        name : string, optional
            Group name

    **Returns**

        out : group
            Group of ones with the given shape and dtype.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.zeros_like` : Return an group of zeros with shape and type of input.
    * :meth:`dana.ones_like` : Return an group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.

    **Examples**

    >>> dana.ones((2,2))
    group([[(1.0, True), (1.0, True)],
           [(1.0, True), (1.0, True)]],
          dtype=[('V', '<f8'), ('mask', '|b1')])
    >>> dana.ones((2,2), dtype=int)
    group([[(1, True), (1, True)],
           [(1, True), (1, True)]],
          dtype=[('V', '<f8'), ('mask', '|b1')])
    '''
    return group(shape=shape,dtype=dtype,keys=keys,
                 mask=mask,name=name,fill=1)


def empty_like(other):
    ''' 
    Returns an unitialized group with the same shape and type as the given group.

    **Parameters**

        other : group_like
            The shape and data-type of other defines the parameters of the
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

    >>> dana.empty_like(numpy.random.random((2,2)))
    group([[(1.0, True), (1.0, True)],
           [(1.0, True), (1.0, True)]], 
          dtype=[('V', '<f8'), ('mask', '|b1')])
    '''
    return group(shape=other,dtype=other.dtype)


def zeros_like(other):
    '''
    Returns a group of zeros with the same shape and type as a given array.

    **Parameters**

        other : group_like
            The shape and data-type of other defines the parameters of the
            returned group.

    **Returns**

        out : group
            Group of zeros with same shape and type as `other`.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.empty` : Return a new uninitialized group.
    * :meth:`dana.ones_like` : Return a group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.

    **Examples**

    >>> dana.zeros_like(numpy.random.random((2,2)))
    group([[(0.0, True), (0.0, True)],
           [(0.0, True), (0.0, True)]], 
          dtype=[('V', '<f8'), ('mask', '|b1')])
    '''
    return group(shape=other,dtype=other.dtype,fill=0)


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
    * :meth:`dana.zeros_like` : Return a group of zeros with shape and type of input.
    * :meth:`dana.empty_like` : Return an empty group with shape and type of input.

    **Examples**

    >>> dana.ones_like(numpy.random.random((2,2)))
    group([[(1.0, True), (1.0, True)],
           [(1.0, True), (1.0, True)]], 
          dtype=[('V', '<f8'), ('mask', '|b1')])
    '''
    return group(shape=other,dtype=other.dtype,fill=1)



def gaussian(shape=(25,25), width=0.5, center=0.0):
    ''' Generate a gaussian of the form g(x) = height*exp(-(x-center)²/width²).

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

