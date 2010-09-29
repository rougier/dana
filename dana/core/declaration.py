#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
import re
from definition import Definition

class DeclarationError(Exception):
    pass

class Declaration(Definition):
    ''' Declaration of type: 'y : dtype' '''
  
    def __init__(self, definition):
        ''' Builds a new Declaration of type: 'y : dtype' '''
        Definition.__init__(self, definition)
        self._parse(definition)


    def _parse(self, definition):
        '''
        Parse definition and check it is a declaration.

        **Parameters**

        definition : str
            Equation definition of the form 'y : dtype'
        '''
        self._definition = definition
        definition = str(definition.replace(' ',''))
        p = re.compile(r'''(?P<y>\w+) (:(?P<dtype>\w+))?''', re.VERBOSE)
        result = p.match(definition)
        if result:
            self._varname = result.group('y')
            self._lhs = self._varname
            self._rhs = ''
            self._definition = None
            self._dtype = result.group('dtype') or 'float'
        else:
            raise DeclarationError, 'Definition is not a declaration'


    def __call__(self):
        '''
        Evaluate declaration (return dtype)
        '''

        return eval(self.dtype)


    def evaluate(self):
        '''
        Evaluate declaration (return dtype)
        '''

        return eval(self.dtype)


    def __repr__(self):
        ''' x.__repr__() <==> repr(x) '''

        classname = self.__class__.__name__
        return "%s('%s : %s')" % (classname, self._varname, self._dtype)


if __name__ == '__main__':
    eq = Declaration('x : float')
