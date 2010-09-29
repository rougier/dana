#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009, 2010 Nicolas Rougier - INRIA - CORTEX Project
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
import inspect
import numpy as np
from numpy import *
import scipy.sparse as sp
from shared_link import shared_link
from sparse_link import sparse_link
from dense_link import dense_link


class group(object):

    def __init__(self, shape=(), dtype=np.double, keys=['V'],
                 mask=True, fill=None, name=''):
        '''
        A group object represents a multidimensional, homogeneous array of
        fixed-size items. An associated data-type object describes the format
        of each element in the group (its byte-order, how many bytes it
        occupies in memory, whether it is an integer, a floating point number,
        or something else, etc.)
  
        Groups should be generally constructed using `empty`, `zeros` or `ones`
        (refer to the See Also section below). The parameters given here refer
        to a low-level method (`group(...)`) for instantiating an array.
      
        Parameters
        ----------

        shape : tuple of ints
            Shape of created group.
        dtype : data-type, optional
            Any object that can be interpreted as a numpy data type.
        keys : list of strings,  optional
            A list of field to be included within group. This requires dtype to
            be an atomic type.
        mask: array_like, optional
            Initial mask of the group indicating defective units.
        fill: array_mask, optional
            Initial fill value for all group fields.

        See Also
        --------
        zeros : Create a group, each element of which is zero.
        ones : Create a group, each element of which is one.
        empty : Create a group, but leave its allocated memory unchanged (i.e.,
                it contains "garbage").
        '''

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
                dtype = d
            else:
                dtype = [(f, dtype) for f in keys]
        else:
            d = [(name,dtype[i]) for i, name in enumerate(dtype.names)]
            dtype = d

        # Check if there is already a mask
        mask_ = False
        for item in dtype:
            name, d = item
            if name == 'mask':
                mask_ = True
        if not mask_:
            dtype.append( ('mask', np.bool8) )

        object.__setattr__(self, '_dtype', np.dtype(dtype))
        object.__setattr__(self, '_shape', shape)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_data', {})
        for key in self._dtype.names:
            self._data[key] = np.ndarray(shape=shape,
                                         dtype=self._dtype[key])
            if fill is not None:
                self._data[key][...] = fill
            object.__setattr__(self,'d'+key, '')
        self.mask[...] = mask
        object.__setattr__(self, '_link', {})
        object.__setattr__(self, '_data_equation', {})
        object.__setattr__(self, '_data_compiled', [])
        object.__setattr__(self, '_link_compiled', [])
        object.__setattr__(self, '_link_equation', {})
        object.__setattr__(self, '_globals', {})


    def reshape(self, shape, order='C'):
        G = group(shape = shape, dtype=self.dtype)
        for key in G._data.keys():
            G._data[key] = self._data[key].reshape(shape)
        G._link = self._link
        G._data_equation = self._data_equation
        G._link_equation = self._link_equation
        G._globals = self._globals
        return G
    reshape.__doc__ = np.reshape.__doc__ 


    def __getattr__(self, key):
        if key in self._data.keys():
            return self._data[key]
        elif key in self._link.keys():
            return self._link[key]
        elif key[0]=='d' and key[1:] in self._data.keys():
            return self._data_equation[key[1:]]
        elif key[0]=='d' and key[1:] in self._link.keys():
            return self._link_equation[key[1:]]
        else:
            return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        if key in self._data.keys():
            self._data[key][...] = value
        elif key[0]=='d' and key[1:] in self._data.keys():
            self._data_equation[key[1:]] = value
            self.compile()
        elif key[0]=='d' and key[1:] in self._link.keys():
            if isinstance(self._link[key[1:]], shared_link):
                raise ValueError, 'Shared link cannot be learned'
            self._link_equation[key[1:]] = value
            self.compile()
        else:
            object.__setattr__(self, key, value)


    def _get_shape(self):
        return self._shape
    def _set_shape(self, shape):
        for key in self._dtype.names:            
            self._data[key].shape = shape
        self._shape = shape
    shape = property(_get_shape, _set_shape,
                     doc=np.ndarray.shape.__doc__)

    def _get_size(self):
        return self._data['mask'].size
    size = property(_get_size,
                    doc=np.ndarray.size.__doc__)

    def _get_dtype(self):
        return self._dtype
    dtype = property(_get_dtype,
                     doc=np.ndarray.dtype.__doc__)

    def __getitem__(self, key):
        ''' x.__getitem__(y) <==> x[y] '''
        return self._data[key]

    def __setitem__(self, key, value):
        ''' x.__setitem__(i, y) <==> x[i]=y '''
        self._data[key][...] = value


    def connect(self, src, kernel, name, dtype=np.double,
                sparse=False, shared=False):
        
        if name[-1] in ['*','-']:
            lname = name[:-1]
        else:
            lname = name
        if sparse or sp.issparse(kernel):
            self._link[lname] = sparse_link(src, self, kernel, name, dtype)
        elif shared:
            if name[-1] != '-':
                self._link[lname] = shared_link(src, self, kernel, name, dtype)
            else:
                raise ValueError, 'Cannot compute distance for shared link'
        else:
            self._link[lname] = dense_link(src, self, kernel, name, dtype)


    def compile(self):

        self._globals = {}
        self._globals['self'] = self
        self._data_compiled = []
        self._link_compiled = []
        for key in self._data_equation.keys():
            self._globals[key] = self[key] #.reshape((self[key].size(),1))
            eqn = self._data_equation.get(key, None)
            if eqn:
                if self.mask.all():
                    expr = compile("%s += %s" % (key, eqn),
                                   "<string>", "exec")
                else:
                    expr = compile("%s += %s; %s *= self.mask" % (key, eqn, key),
                                   "<string>", "exec")
                self._data_compiled.append(expr)

                # Get unknown constants (and only them) from upper frame
                for i in range(1,len(inspect.stack())):
                    frame = inspect.stack()[i][0]
                    for name in expr.co_names:
                        if (name in frame.f_globals.keys() and
                            name not in self._globals      and
                            name not in self._data_equation.keys()):
                            self._globals[name] = frame.f_globals[name]

        for key in self._link.keys():
            link = self._link[key]
            link.compile()

        for key in self._link_equation.keys():
            eqn = self._link_equation.get(key, None)
            if eqn:
                link = self._link[key]
                src, dst = link._src, link._dst
                _locals = {}
                _locals['pre']  = src.reshape((1,src.size))
                _locals['post'] = dst.reshape((dst.size,1))
                _locals['W'] = link._kernel
                if isinstance(link,dense_link):
                    _locals['kernel'] = self._link[key]._kernel
                    _locals['mask'] = self._link[key]._mask
                    if self._link[key]._mask.all():
                        expr = compile("W += %s" % eqn,
                                       "<string>", "exec")
                    else:
                        expr = compile("W += %s; kernel *= mask" % eqn,
                                       "<string>", "exec")
                    self._link_compiled.append((key, expr, _locals))
                elif isinstance(link,sparse_link):
                    expr = compile("W += %s" % eqn, "<string>", "exec")
                    self._link_compiled.append((key, expr, _locals))

                # Get unknown constants (and only them) from upper frame
                for i in range(1,len(inspect.stack())):
                    frame = inspect.stack()[i][0]
                    for name in expr.co_names:
                        if (name not in ['W', 'pre', 'post'] and
                            name in frame.f_globals.keys() and
                            name not in self._globals      and
                            name not in self._data_equation.keys()):
                            self._globals[name] = frame.f_globals[name]


    def compute(self, dt=0.1):
        self._globals['dt'] = dt
        for key in self._link.keys():
            self._globals[key] = self._link[key].compute()
        for expr in self._data_compiled:
            exec(expr, self._globals)
        for key in self._link.keys():
            dst = self._link[key]._dst
            self._globals[key] = self._globals[key].reshape((dst.size,1))

    def learn(self, dt=0.1):
        self._globals['dt'] = dt
        for item in self._link_compiled:
            key, expr, _locals = item
            exec(expr, self._globals, _locals)

    def get_weight(self, src, key):
        ''' Return the connection array from source to self[key] 
        
        **Parameters**
        src : group
            Source group to be considered
        key: tuple of integers
            Position within group to be considered

        **Returns**
        out: array
            Connection array linking self[key] to source
        '''

        for link in self._link.values():
            if id(src) == id(link._src):
                return link[key]
        return np.ones(src.shape)*np.NaN


    def asarray(self):
        ''' Return a copy of self as a regular numpy interleaved array without mask. '''

        dtype=[]
        for key in self.dtype.names:
            if key != 'mask':
                dtype.append( (key,self.dtype[key]))
        dtype = np.dtype(dtype)
        R = np.zeros(shape=self.shape, dtype=dtype)
        for key in self._data.keys():
            if key != 'mask':
                R[key] = self._data[key]*self.mask
        return R
