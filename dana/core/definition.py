#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Generic definition of type:

* :class:`DifferentialEquation` (``dY/dt = expr : type``)
* :class:`Equation` (``Y = expr : type``)
* :class:`Declaration` (``Y : type``)
'''

class DefinitionError(Exception):
    ''' Definition Error '''
    pass


class Definition(object):
    ''' Generic definition of type:

    * :class:`DifferentialEquation` (``dY/dt = expr : type``)
    * :class:`Equation` (``Y = expr : type``)
    * :class:`Declaration` (``Y : type``)
    '''
  
    def __init__(self, definition):
        self._definition = None
        self._varname = None
        self._dtype = None

    def _parse(self, definition):
        ''' Parse definition '''
        raise NotImplemented(definition)

    def __repr__(self):
        ''' x.__repr__() <==> repr(x) '''

        classname = self.__class__.__name__
        return "%s('%s = %s : %s')" % (classname, self._lhs, self._rhs, self._dtype)


    def _get_varname(self):
        ''' Get variable name (left hand side) '''
        return self._varname
    varname = property(_get_varname,
                       doc='''Equation variable name (left hand side) ''')

    def _get_lhs(self):
        ''' Get equation left hand side '''
        return self._lhs
    lhs = property(_get_lhs,
                   doc='''Equation left hand-side''')

    def _get_rhs(self):
        ''' Get equation right hand side '''
        return self._rhs
    rhs = property(_get_lhs,
                   doc='''Equation right hand-side''')

    def _get_definition(self):
        ''' Get equation original definition '''
        return self._definition
    definition = property(_get_definition,
                          doc='''Equation original definition''')

    def _get_dtype(self):
        '''Get equation data type '''
        return self._dtype
    dtype = property(_get_dtype,
                     doc='''Equation data type''')


