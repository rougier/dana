#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
A group object represents a multidimensional, homogeneous group of contiguous
numpy arrays.  An associated data-type object describes the format of each
element in the group (its byte-order, how many bytes it occupies in memory,
whether it is an integer or a floating point number, etc.).

A group is very similar to a numpy record array and those not familiar 
should have a look at numpy first.
'''
import numpy

class group(object):
    '''
    A group object represents a multidimensional, homogeneous group of
    contiguous numpy arrays.  An associated data-type object describes the
    format of each element in the group (its byte-order, how many bytes it
    occupies in memory, whether it is an integer or a floating point number,
    etc.).

    A group is very similar to a numpy record array and those not familiar 
    should have a look at numpy first.

    **See also**

    * :meth:`zeros` : Return a new group setting values to zero.
    * :meth:`ones` : Return a new group setting values to one.
    * :meth:`empty` : Return an unitialized group.
    * :meth:`zeros_like` : Return a group of zeros with shape and type of input.
    * :meth:`ones_like` : Return a group of ones with shape and type of input.
    * :meth:`empty_like` : Return a empty group with shape and type of input.
    '''

    def __init__(self, shape=(), dtype=float, order='C', fill=None):
        '''
        Creates a new group

        Groups should be constructed using `ones`, `zeros` or `empty` (refer to
        the ``See also`` section above). The parameters given here describe a
        low-level method for instantiating a group.
      
        **Parameters**

        shape : tuple of ints
            Shape of created group.

        dtype : data-type, optional
            Any object that can be interpreted as a numpy data type.

        order : {'C', 'F'}, optional
            Row-major or column-major order.

        fill : scalar
            Fill value to be used to fill group fields
        '''
        if type(shape) is int:
            shape = (shape,)
        elif type(shape) is numpy.ndarray:
            obj = shape
            shape = obj.shape
            dtype = obj.dtype
            if fill is None:
                fill = obj
        if not isinstance(dtype, numpy.dtype):
            if type(dtype) == list:
                d = [(n, t) for n, t in dtype]
                dtype = d
            else:
                dtype = [('f0', numpy.dtype(dtype)), ]
        elif dtype.type is numpy.void:
            d = [(name, dtype[i]) for i, name in enumerate(dtype.names)]
            dtype = d
        else:
            dtype = [('f0', numpy.dtype(dtype)), ]
        object.__setattr__(self, '_dtype', numpy.dtype(dtype))
        object.__setattr__(self, '_shape', shape)
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_base', None)
        object.__setattr__(self, '_scalar', None)
        object.__setattr__(self, '_order', order)
        object.__setattr__(self, '_keys', numpy.dtype(dtype).names)
        for key in self._keys:
            self._data[key] = numpy.empty(shape=shape,
                                          dtype=self._dtype[key],
                                          order=order)
            if fill is not None:
                if type(fill) in [bool, int, float]:
                    self._data[key][...] = fill
                elif type(fill) is numpy.ndarray:
                    self._data[key][...] = fill[key] 
        if type(fill) in [tuple, list]:
            self[...] = fill


    def _get_data(self):
        '''Get group data'''
        return self._data
    data = property(_get_data,
                    doc='''Group data (list of arrays)''')

    def _get_keys(self):
        '''Get group keys'''
        return self._keys
    keys = property(_get_keys,
                    doc='''Group keys''')


    def item(self):
        '''
        Copy the first element of group to a standard Python scalar and return
        it. The group must be of size one.
        '''
        return self._data[self._data.keys()[0]]


    def flatten(self):
        return self.reshape( (self.size,) )

    def reshape(self, shape):
        '''
        Gives a new shape to the group without changing its data.

        **Parameters**

        shape : {tuple, int}
            The new shape should be compatible with the original shape. If
            an integer, then the result will be a 1-D group of that length.
            One shape dimension can be -1. In this case, the value is inferred
            from the length of the array and remaining dimensions.

        **Returns**

        reshaped_group : group
            This will be a new view object if possible; otherwise, it will
            be a copy.

        **Examples**

        >>> g = group([[1,2,3], [4,5,6]])
        >>> g.reshape(6)
        group([1, 2, 3, 4, 5, 6])
        '''
        G = group(shape=(), dtype=self.dtype)
        for key in G.keys:
            G.data[key] = self.data[key].reshape(shape)
        G._shape = shape
        return G


    def __len__(self):
        ''' x.__len__() <==> len(x) '''
        if self.shape:
            return self.shape[0]
        raise TypeError, 'len() of unsized object'


    def __getattr__(self, key):
        ''' '''
        if key in self._keys:
            return self._data[key]
        else:
            return object.__getattribute__(self, key)


    def __setattr__(self, key, value):
        ''' '''
        if key in self._keys:
            self._data[key][...] = value
        else:
            object.__setattr__(self, key, value)


    def __call__(self, keys):
        ''' '''
        return self.subgroup(keys)


    def subgroup(self, key):
        ''' '''
        dtype = []
        dtype.append( (key, self._dtype[key]) )
        G = group(shape=self.shape, dtype=dtype)
        G.data[key] = self.data[key]
        G._base = self
        return G


    def _get_shape(self):
        '''Get group shape'''
        return self._shape
    def _set_shape(self, shape):
        '''Set group shape'''
        for key in self._dtype.names:            
            self._data[key].shape = shape
        self._shape = shape
    shape = property(_get_shape, _set_shape,
                     doc='''Tuple of group dimensions.''')


    def _get_size(self):
        '''Get group size'''
        return self._data.values()[0].size
    size = property(_get_size,
                    doc = '''Number of elements in the group.''')


    def _get_base(self):
        '''Get group base'''
        return self._base
    base = property(_get_base,
                    doc = '''Base group.''')


    def _get_dtype(self):
        '''Get group dtype'''
        return self._dtype
    dtype = property(_get_dtype,
                     doc='''Data-type for the group.''')


    def __getitem__(self, key):
        if type(key) is str:
            if key in self._keys:
                return self._data[key]
            else:
                raise ValueError, 'field named %s not found' % key
        elif type(key) in [int, slice, tuple]:
            shape = self._data.values()[0][key].shape
            if shape is not ():
                G = group(shape, self._dtype)
                for name in self._dtype.names:
                    G.data[name] = self.data[name][key]
                return G
            elif len(self.data) == 1:
                return self.data.values()[0][key]
            else:
                return tuple(self.data[k][key] for k in self._keys)

        elif key is Ellipsis:
            return self
        elif not len(self._shape):
            if key is Ellipsis:
                return self
            if type(key) is str:
                raise ValueError, 'field named %s not found' % key
            elif type(key) is slice:
                raise ValueError, 'cannot slice a 0-d group'
            elif type(key) is tuple:
                raise IndexError, \
                    '''0-d groups can only use a single () or a ''' \
                    '''list of newaxes (and a single ...) as an index'''
            else:
                raise IndexError, "0-d groups can't be indexed"
        raise IndexError, 'index must be either an int or a sequence'


    def __setitem__(self, key, value):
        if type(key) is str:
            if key in self._keys:
                self._data[key][...] = value
                return
            elif type(value) in [int, float, bool]:
                Z = numpy.ones(shape=self._shape, dtype=type(value),
                               order=self._order)*value
                self._data[key] = Z
                dtype = [(name, self.dtype[i])
                         for i, name in enumerate(self.dtype.names)]
                dtype.append((key, Z.dtype))
                self._dtype = numpy.dtype(dtype)
                self._keys = numpy.dtype(dtype).names
                return
            elif type(value) is numpy.ndarray:
                if value.size == self.size and \
                        value.dtype.names == None:
                    self._data[key] = value.reshape(self.shape)
                    dtype = [(name, self.dtype[i])
                             for i, name in enumerate(self.dtype.names)]
                    dtype.append((key, value.dtype))
                    self._dtype = numpy.dtype(dtype)
                    self._keys = numpy.dtype(dtype).names
                    return
                elif value.dtype.names is not None:
                    raise ValueError, \
                        "Data cannot be a record array"
                else:
                    raise ValueError, \
                        "Data size must match group size"
            else:
                raise ValueError, \
                    "Data-type not understood"                        
        elif type(key) in [int, slice, tuple] or key is Ellipsis:
            if key is Ellipsis:
                G = self
            else:
                G = self.__getitem__(key)
            if type(G) is group:
                if type(value) in [bool, int, float]:
                    for k in self._keys:
                        G.data[k][...] = value
                    return
                elif type(value) in [tuple, list]:
                    if len(value) == len(self._keys):
                        for i, k in enumerate(self._keys):
                            G.data[k][...] = value[i]
                        return
                    else:
                        raise ValueError, \
                            'size of tuple must match number of fields.'
                else:
                    raise ValueError, \
                        "Data type not understood"
            elif type(G) is tuple:
                if type(value) in [bool, int, float]:
                    for k in self._keys:
                        self._data[k][key] = value
                    return
                elif type(value) is tuple:
                    if len(value) == len(self._keys):
                        for i, k in enumerate(self._keys):
                            self._data[k][key] = value[i]
                        return
                    else:
                        raise ValueError, \
                            'size of tuple must match number of fields.'
        raise IndexError, 'index must be either an int or a sequence'
 

    def __delitem__(self, key):
        if type(key) is not str:
            raise ValueError, 'key must be a string'
        if key not in self._keys:
            raise ValueError, \
                "field named '%s' does not exist" % key
        del self._data[key]        
        dtype = []
        for i, name in enumerate(self.dtype.names):
            if name != key:
                dtype.append((name, self.dtype[i]))
        self._dtype = numpy.dtype(dtype)
        self._keys = numpy.dtype(dtype).names


    def asarray(self):
        ''' Return a ndarray copy of this group '''
        return numpy.array(self, dtype=self.dtype)


    def __str__(self):
        ''' x.__str__() <==> str(x) '''
        return numpy.array_str(self)


    def __repr__(self):
        ''' x.__repr__() <==> repr(x) '''
        return numpy.array_repr(self)
