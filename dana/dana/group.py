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
from array import array
from numpy import minimum, maximum, sin, cos, exp, sqrt, multiply, dot
from functions import extract, convolve1d, convolve2d
import link
import inspect


class group(object):
    ''' A group represents a vector of homogeneous elements.

    Elements can have an arbitrary number of values whose name and type must be
    described at group creation. These values can be later accessed using either
    dictionary lookups such as ``group['x']`` and ``group['y']``.

    **Examples**

    Create a group with two values, ``x`` and ``y``:

    >>> G = group((2,2), keys=['x','y'])
    >>> G
    group([[(0.0, 0.0, True), (0.0, 0.0, True)],
           [(0.0, 0.0, True), (0.0, 0.0, True)]], 
           dtype=[('x', '<f8'), ('y', '<f8'), ('mask', '|b1')])
    >>> G['x']
    group([[ 0.,  0.],
           [ 0.,  0.]])
    '''

    def __init__(self, shape=(), dtype=np.float32, keys=['V'],
                 mask=True, name='', fill=None):
        ''' Create a group.
        
        **Parameters**

            shape : tuple of integer or array-like object
                Shape of output array or object to create group from.
            dtype : data-type
                Desired data-type.
            keys : list of strings
                Names of the different values
            mask : boolean or boolean array
                boolean mask indicating active and inactive elements
            name : string
                Group name
            fill : scalar or scalar array
                Fill value

        **Returns**
            out: group
                Group of given shape and type.
        '''

        object.__setattr__(self,'_values', {})
        object.__setattr__(self,'_equations', {})
        object.__setattr__(self,'_links', {})
        object.__setattr__(self,'_stored', {})
        
        if type(shape) is list:
            Z = np.array(shape)
            if fill is None:
                fill = Z
            dtype = [(f, Z.dtype) for f in keys]
            shape = Z.shape
        elif type(shape) is np.ndarray:
            if fill is None:
                fill = shape
            dtype = [(f, shape.dtype) for f in keys]
            shape = shape.shape
        elif type(shape) is int:
            shape = (shape,)
        if not isinstance(dtype,np.dtype):
            if type(dtype) == list:
                d = [(n,t) for n,t in dtype]
                d += [('mask',np.bool)]
                dtype = d
            else:
                dtype = [(f, dtype) for f in keys]
                dtype += [('mask',np.bool)]
        else:
            d = [(name,dtype[i]) for i, name in enumerate(dtype.names)]
            d += [('mask',np.bool)]
            dtype = d
        self._dtype = np.dtype(dtype)
        for i in range(len(self._dtype)):
            dtype, key = self._dtype[i], self._dtype.names[i]
            self._values[key] = array(shape=shape, dtype=dtype, parent=self)
            if fill is not None and key != 'mask':
                self._values[key][...] = fill
        self['mask'] = mask
        self.name = name


    def __getattr__(self, key):
        if key in self._values.keys():
            return self._values[key]
        elif key[0]=='d' and key[1:] in self._values.keys():
            return self._equations[key[1:]]
        elif key in self._links.keys():
            return self._links[key]
        else:
            return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        if key in self._values.keys():
            self._values[key][...] = value
        elif key[0]=='d' and key[1:] in self._values.keys():
            self._equations[key[1:]] = value
        elif key[0]=='d' and key[1:] in self._links.keys():
            if self._links[key[1:]].shared:
                raise ValueError, 'Shared link cannot be learned'
            self._equations[key[1:]] = value
        else:
            object.__setattr__(self, key, value)

    def asarray(self):
        ''' Return a copy of self as a regular numpy interleaved array without mask. '''
        dtype=[]
        for key in self.dtype.names:
            if key != 'mask':
                dtype.append( (key,self.dtype[key]))
        dtype = np.dtype(dtype)
        R = np.zeros(shape=self.shape, dtype=dtype)
        for key in self._values.keys():
            if key != 'mask':
                R[key] = self[key]*self['mask']
        return R

    def __repr__(self):
        ''' Return a string representation of group '''
        dtype=[]
        for key in self.dtype.names:
            dtype.append( (key,self.dtype[key]))
        dtype = np.dtype(dtype)
        R = np.zeros(shape=self.shape, dtype=dtype)
        for key in self._values.keys():
            R[key] = self[key]*self['mask']
        return repr(R).replace('array','group')

    def __getitem__(self, key):
        if type(key) is str:
            return self._values[key]
        else:
            return self.asarray()[key]

    def __setitem__(self, key, value):
        #if type(key) is str:
        self._values[key][...] = value
        #else:
        #    self._values['mask'][key] = value

    def _get_shape(self):
        return self._values[self._values.keys()[0]].shape

    def _set_shape(self, shape):
        for key in self._values.keys():
            #self._values[key].shape = shape
            self._values[key]._force_shape(shape)
            #self._values[key].reshape(shape)
    shape = property(_get_shape, _set_shape,
                     doc = '''Tuple of group dimensions.\n
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


    def reshape(self, shape):
        G = group(shape)
        G.name = self.name
        G._links = self._links
        G._equations = self._equations
        G._dtype = self._dtype
        for key in self._values.keys():
            G._values[key] = self._values[key].reshape(shape)
        return G

    def _get_dtype(self):
        return self._dtype
    def _set_dtype(self):
        raise AttributeError, \
            '''attribute 'dtype' of 'group' objects is not writable'''
    dtype = property(_get_dtype, _set_dtype,
                     doc = 'Data-type for the group')


    def _get_size(self):
        return self._values[self._values.keys()[0]].size
    def _set_size(self):
        raise AttributeError, \
            '''attribute 'size' of 'group' objects is not writable'''
    size = property(_get_size, _set_size,
                     doc = '''Number of elements in the group.\n
                              **Examples**

                              >>> x = dana.zeros((3,5,2), dtype=int)
                              >>> x.size
                              30''')



    def connect(self, source, kernel, name, dtype=np.float64, sparse=None, shared=False):
        ''' Connect group to source group 'source' using 'kernel'.

        **Parameters**

            source : group
                Source group.

            kernel : array or sparse array
                Kernel array to be used for linking source to group.

            name : string
                Name of the link to be made. 

            dtype : data-type
                The desired data-type.
            
            sparse: True | False | None
                Indicate wheter internal storage should be sparse

            shared: True or False
                Whether the kernel is shared among elements composing the group.
                (only available for one-dimensional and two-dimensional groups)
        '''
        
        if name[-1] not in ['*','-']:
            lname = name
            name += '*'
        else:
            lname = name[:-1]
        self._links[lname] = link.link(source=source, destination=self,
                                       kernel=kernel, name=name, dtype=dtype,
                                       sparse=sparse, shared=shared)


    def disconnect(self, name):
        ''' Disconnect a named link from source.
        
        **Parameters**

            name : string
                Name of the link to be removed.
        '''

        if name in self._links.keys():
            del self._links[name]



    def compute(self, dt=0.1):
        ''' Compute new values according to equations.

        **Parameters**

            dt : float
                Time period to consider
        '''

        # Get relevant namespaces
        frame=inspect.stack()[1][0]
        f_globals, f_locals = frame.f_globals,frame.f_locals
        namespace = globals()
        namespace.update(f_globals)
        namespace.update(f_locals)
        namespace['dt'] = dt

        # Get activities from all fields
        for key in self._values.keys():
            namespace[key] = self[key]

        # Compute links weighted sums
        self._stored = {}
        for key in self._links.keys():
            namespace[key] = self._links[key].compute()
            self._stored[key] = namespace[key]

        # Evaluate equations for each field
        dV = []
        for key in self._values.keys():
            if key in self._equations.keys() and self._equations[key]:
                result = eval(self._equations[key], f_globals, namespace)
                if result.__class__.__name__ == 'matrix':
                    result = np.array(result).reshape(self[key].shape)
                self[key] = self[key] + np.multiply(result,self['mask'])
                if type(result) not in [float,int]:
                    dV.append(result.flatten().sum())
                else:
                    dV.append(result)
        return dV


    def learn (self, dt=0.1, namespace=globals()):
        ''' Adapt group links according to equations

        **Parameters**

            dt : float
                Time period to consider
        '''

        # Get relevant namespaces
        frame=inspect.stack()[1][0]
        f_globals, f_locals = frame.f_globals,frame.f_locals
        namespace = globals()
        namespace.update(f_globals)
        namespace.update(f_locals)
        namespace['dt'] = dt

        links = {}
        for key in self._stored.keys():
            links[key] = self._stored[key]
        # Evaluate equations for each link
        for key in self._links.keys():
            if key in self._equations.keys():
                self._links[key].learn(self._equations[key], links, dt, namespace)



    def get_weight(self, source, key):
        ''' Return the connection array from source to self[key] 
        
        **Parameters**

            source : group
                Source group to be considered

            key: tuple of integers
                Position within group to be considered

        **Returns**

            out: array
                Connection array linking self[key] to source
        '''

        Z = np.ones(source.shape)*np.NaN
        src = source.parent
        for k in self._links.keys():
            L = self._links[k]
            lsrc = L.source.parent
            if id(src) == id(lsrc):
                Z = L[key]
                break
        return Z
