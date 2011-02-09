#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
import inspect
import numpy as np
from model import Model
from network import __default_network__
from definition import Definition, DefinitionError
from declaration import Declaration, DeclarationError
from diff_equation import DifferentialEquation, DifferentialEquationError


class GroupError(Exception):
    pass


class Group(object):
    '''
    A group object represents a multidimensional, homogeneous group of
    contiguous numpy arrays.  An associated data-type object describes the
    format of each element in the group (its byte-order, how many bytes it
    occupies in memory, whether it is an integer or a floating point number,
    etc.).

    A group is very similar to a numpy record array and those not familiar 
    should have a look at numpy first.

    **See also**

    * :meth:`dana.zeros` : Return a new group setting values to zero.
    * :meth:`dana.ones` : Return a new group setting values to one.
    * :meth:`dana.empty` : Return an unitialized group.
    * :meth:`dana.zeros_like` : Return a group of zeros with shape and type of input.
    * :meth:`dana.ones_like` : Return a group of ones with shape and type of input.
    * :meth:`dana.empty_like` : Return a empty group with shape and type of input.
    '''

    def __init__(self, shape=(), dtype=float, model=None, fill=0.0):
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

        model : [str | :class:'~dana.Model']
            Set of equations describing group behavior

        fill : scalar
            Fill value to be used to fill group fields
        '''

        # Model is prevalent over dtype
        if model is not None or (type(dtype) is str and model is None):
            if type(model) is str:
                model = Model(model)
            elif type(dtype) is str and model is None:
                model = Model(dtype)
            dtype = []
            for eq in model._diff_equations:
                dtype.append((eq._varname, eq._dtype))
            for eq in model._equations:
                dtype.append((eq._varname, eq._dtype))
            for eq in model._declarations:
                dtype.append((eq._varname, eq._dtype))
        else:
            model = Model('')

        if type(shape) is int:
            shape = (shape,)
        elif type(shape) is np.ndarray:
            obj = shape
            shape = obj.shape
            dtype = obj.dtype
            if fill is None:
                fill = obj
        if not isinstance(dtype, np.dtype):
            if type(dtype) == list:
                d = [(n, t) for n, t in dtype]
                dtype = d
            else:
                dtype = [('f0', np.dtype(dtype)), ]
        elif dtype.type is np.void:
            d = [(name, dtype[i]) for i, name in enumerate(dtype.names)]
            dtype = d
        else:
            dtype = [('f0', np.dtype(dtype)), ]

        object.__setattr__(self, '_dtype', np.dtype(dtype))
        object.__setattr__(self, '_shape', shape)
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_base', None)
        object.__setattr__(self, '_scalar', None)
        object.__setattr__(self, '_keys', np.dtype(dtype).names)                
        object.__setattr__(self, '_connections', [])
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_namespace', {})

        for key in self._keys:
            self._data[key] = np.empty(shape=shape,
                                          dtype=self._dtype[key])
            if fill is not None:
                if type(fill) in [bool, int, float]:
                    self._data[key][...] = fill
                elif type(fill) is np.ndarray:
                    self._data[key][...] = fill[key] 
        if type(fill) in [tuple, list]:
            self[...] = fill
    
        __default_network__.append(self)
        try:
           self.setup()
        except:
            pass


    def setup(self, namespace=None):
        ''' '''
        
        # Get unknown constants (and only them) from upper frame
        if not namespace:
            namespace = {}
        numpy_ns = dir(np)
        unknown = {}
        for eq in self._model._diff_equations:
            namespace[eq._varname] = self[eq._varname]
        for eq in self._model._equations:
            namespace[eq._varname] = self[eq._varname]
        for eq in self._model._declarations:
            namespace[eq._varname] = self[eq._varname]
        for eq in self._model._diff_equations:
            for var in eq._variables:
                if var not in numpy_ns and var not in namespace \
                        and var not in unknown:
                    unknown[var] = None
        for eq in self._model._equations:
            for var in eq._variables:
                if var not in numpy_ns and var not in namespace \
                        and var not in unknown:
                    unknown[var] = None
        for i in range(1,len(inspect.stack())):
            frame = inspect.stack()[i][0]
            for name in unknown:
                if name in frame.f_globals.keys() and name not in namespace:
                    unknown[name] =  frame.f_globals[name]
                    namespace[name] =  frame.f_globals[name]
        self._namespace = namespace
        self._namespace.update(globals())

        # Make sure all masked units are set to 0
        if hasattr(self,'mask'):
            for key in self._data.keys():
                self._data[key] *= self.mask

        # Make sure all masked connections source units are set to 0
        for connection in self._connections:
            source = connection._source
            if hasattr(source,'mask'):
                for key in source._data.keys():
                    source._data[key] *= source.mask


    def evaluate(self, dt=1):
        ''' '''

        self._namespace['dt'] = dt
        saved = {}

        for connection in self._connections:
            connection.propagate()

        # Differential equations
        for eq in self._model._diff_equations:
            saved[eq._varname] = self[eq._varname].copy()
            self._namespace[eq._varname] = self[eq._varname]
        for eq in self._model._diff_equations:
            args = [saved[eq._varname],dt]+ \
                   [self._namespace[var] for var in eq._variables]
            eq.evaluate(*args)
        for eq in self._model._diff_equations:
            self[eq._varname][...] = saved[eq._varname]

        # Equations
        for eq in self._model._equations:
            saved[eq._varname] = self[eq._varname].copy()
            self._namespace[eq._varname] = self[eq._varname]
        for eq in self._model._equations:
            args = [self._namespace[var] for var in eq._variables]
            saved[eq._varname][...] = eq.evaluate(*args) #*self._mask
        for eq in self._model._equations:
            self[eq._varname][...] = saved[eq._varname]

        # Make sure all masked units are set to 0
        if hasattr(self,'mask'):
            for key in self._data.keys():
                self._data[key] *= self.mask

        # Learning
        for connection in self._connections:
            connection.evaluate(dt)


    def run(self, t=1.0, dt=0.01, n=None):
        ''' '''

        if n == None:
            n = int(t/dt)
        else:
            dt = 1
        self.setup()
        args,saved = {}, {}

        for eq in self._model._diff_equations:
            args[eq.varname] = [self._data[eq.varname],dt]+ \
                [self._namespace[var] for var in eq._variables]
        for eq in self._model._equations:
            args[eq.varname] = [self._namespace[var] for var in eq._variables]
        self._namespace['dt'] = dt

        for i in range(int(t/dt)):
            for connection in self._connections:
                connection.propagate()
            for eq in self._model._diff_equations:
                eq.evaluate(*args[eq.varname])
            for eq in self._model._equations:
                self._data[eq.varname][...] = eq.evaluate(*args[eq.varname])

            # Make sure all masked units are set to 0
            if hasattr(self,'mask'):
                for key in self._data.keys():
                    self._data[key][...] *= self.mask

            for connection in self._connections:
                connection.evaluate(dt)


    def item(self):
        '''
        Copy the first element of group to a standard Python scalar and return
        it. The group must be of size one.
        '''
        return self._data[self._data.keys()[0]]

 

    def ravel(self):
        '''
        Return a flattened group.
    
        A 1-D group, containing the elements of the group, is returned.
        '''
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
        G = Group(shape=(), dtype=self.dtype)
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
        G = Group(shape=self.shape, dtype=dtype)
        G.data[key] = self.data[key]
        G._base = self.base or self

        # Get subgroup relevant connections
        base = G._base
        model = base.model
        deps, done, exts = [key], [], []
        while deps:
            var = deps[0]
            deps.remove(var)
            done.append(var)
            if var not in model.variables:
                continue
            if isinstance(model[var], Declaration):
                if var not in exts: exts.append(var)
                continue
            eq = model[var]
            for v in eq._variables:
                if v not in base.keys:
                    continue
                elif isinstance(model[v], Declaration):
                    if v not in exts: exts.append(v)
                    continue
                elif v not in done and v not in deps:
                    deps.append(v)
        for connection in base.connections:
            if connection.target_name in exts:
                G.connections.append(connection)

        return G


    def __getitem__(self, key):
        ''' '''

        if type(key) is str:
            if key in self._keys:
                return self._data[key]
            else:
                raise ValueError, 'field named %s not found' % key

        elif type(key) in [int, slice, tuple]:
            shape = self._data.values()[0][key].shape
            if shape is not ():
                G = Group(shape, self._dtype)
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
                Z = np.ones(shape=self._shape, dtype=type(value))*value
                self._data[key] = Z
                dtype = [(name, self.dtype[i])
                         for i, name in enumerate(self.dtype.names)]
                dtype.append((key, Z.dtype))
                self._dtype = np.dtype(dtype)
                self._keys = np.dtype(dtype).names
                return
            elif type(value) is np.ndarray:
                if value.size == self.size and \
                        value.dtype.names == None:
                    self._data[key] = value.reshape(self.shape)
                    dtype = [(name, self.dtype[i])
                             for i, name in enumerate(self.dtype.names)]
                    dtype.append((key, value.dtype))
                    self._dtype = np.dtype(dtype)
                    self._keys = np.dtype(dtype).names
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
            if type(G) is Group:
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
        self._dtype = np.dtype(dtype)
        self._keys = np.dtype(dtype).names


    def asarray(self):
        ''' Return a ndarray copy of this group '''

        return np.array(self, dtype=self.dtype)


    def as_masked_array(self):
        ''' Return a masked ndarray copy of this group '''

        if hasattr(self,'mask') and self.mask != None:
            mask = self.mask
        else:
            mask = 1
        Z = np.ma.array(self, dtype=self.dtype)
        dtype = []
        for i, name in enumerate(self.dtype.names):
            dtype.append((name, int))
        Z.mask = np.ones(self.shape,dtype=dtype)
        for i, name in enumerate(self.dtype.names):
            Z.mask[name] = 1-self.mask
        return Z


    def __str__(self):
        ''' x.__str__() <==> str(x) '''
        if hasattr(self,'mask') and self.mask != None:
            return str(self.as_masked_array())
        else:
            return np.array_str(self)


    def __repr__(self):
        ''' x.__repr__() <==> repr(x) '''
        if hasattr(self,'mask') and self.mask != None:
            return repr(self.as_masked_array())
        else:
            return np.array_repr(self)


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


    def _get_connections(self):
        '''Get group connections'''
        return self._connections
    connections = property(_get_connections,
                           doc='''Group connections''')


    def _get_model(self):
        '''Get group model'''
        return self._model
    model = property(_get_model,
                    doc='''Group model''')
