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
import scipy.sparse as sp
from core import *
from network import __default_network__


class GroupError(Exception):
    pass


class Group(group):
    '''
    A group object represents a multidimensional, homogeneous group of
    contiguous numpy arrays.  An associated data-type object describes the
    format of each element in the group (its byte-order, how many bytes it
    occupies in memory, whether it is an integer or a floating point number,
    etc.).

    Groups should be constructed using `ones`, `zeros` or `empty` (refer to
    the ``See Also`` section below).  The parameters given here describe a
    low-level method for instantiating a group.
    '''

    def __init__(self, shape=(), model=None, fill=0.0):
        '''
        '''

        if type(model) is str:
            model = Model(model)
        dtype = []
        for eq in model._diff_equations:
            dtype.append((eq._varname, eq._dtype))
        for eq in model._equations:
            dtype.append((eq._varname, eq._dtype))
        for eq in model._declarations:
            dtype.append((eq._varname, eq._dtype))
        group.__init__(self,shape, dtype, fill = fill)
        object.__setattr__(self, '_connections', [])
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_namespace', {})
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
    
    def evaluate(self, dt=1):
        ''' '''

        self._namespace['dt'] = dt
        saved = {}


        for connection in self._connections:
            connection.propagate()

        # Differential equations
        for eq in self._model._diff_equations:
            saved[eq._varname] = self[eq._varname].copy()
            self._namespace[eq._varname] = self[eq._varname] #.copy()
        for eq in self._model._diff_equations:
            args = [saved[eq._varname],dt]+ \
                   [self._namespace[var] for var in eq._variables]
            eq.evaluate(*args)
        for eq in self._model._diff_equations:
            self[eq._varname][...] = saved[eq._varname]

        # Equations
        for eq in self._model._equations:
            saved[eq._varname] = self[eq._varname].copy()
            self._namespace[eq._varname] = self[eq._varname] #.copy()
        for eq in self._model._equations:
            args = [self._namespace[var] for var in eq._variables]
            saved[eq._varname][...] = eq.evaluate(*args) #*self._mask
        for eq in self._model._equations:
            self[eq._varname][...] = saved[eq._varname]

        for connection in self._connections:
            connection.evaluate(dt)


    def run(self, t=1, dt=0.1):
        ''' '''

        self.setup()
        args = {}
        for eq in self._model._diff_equations:
            args[eq.varname] = [self._data[eq.varname],dt]+ \
                [self._namespace[var] for var in eq._variables]
        for eq in self._model._equations:
            args[eq.varname] = [self._namespace[var] for var in eq._variables]

        for i in range(int(t/dt)):
            self._namespace['dt'] = dt
            for connection in self._connections:
                connection.propagate()
            for eq in self._model._diff_equations:
                eq.evaluate(*args[eq.varname])
            for eq in self._model._equations:
                self._data[eq.varname][...]= eq.evaluate(*args[eq.varname])
            for connection in self._connections:
                connection.evaluate(dt)


    def _get_model(self):
        '''Get group model'''
        return self._model
    model = property(_get_model,
                    doc='''Group model''')
