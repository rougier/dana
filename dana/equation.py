#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
'''
Equation class.

The Equation class allows to manipulate equations of the type:

   y = expr : dtype

where `y` is the equation variable, `expr` is a valid python expression and
`dtype` describes the type of the y variable. When an equation is created, the
given definition is parsed to check if is of the right form. If this is not the
case, an `EquationError` is raised.

**Examples**

>>> eq = Equation('y = a+b*x')
>>> y = eq.evaluate(1,2,3)             # a=1, b=2, x=3
>>> y = eq.evaluate(x=1, b=2, a=3)     # a=3, b=2, x=1
'''
import re
import numpy as np
import inspect
from definition import Definition

class EquationError(Exception):
    pass

class Equation(Definition):
    ''' Equation of type: 'y = expr : dtype'

    The Equation class allows to manipulate equations of the type:

        y = expr : dtype

    where `y` is the equation variable, `expr` is a valid python expression and
    `dtype` describes the type of the y variable. When an equation is created,
    the given definition is parsed to check if is of the right form. If this is
    not the case, an `EquationError` is raised.

    **Examples**

    >>> eq = Equation('y = a+b*x')
    >>> y = eq.evaluate(1,2,3)             # a=1, b=2, x=3
    >>> y = eq.evaluate(x=1, b=2, a=3)     # a=3, b=2, x=1
    ''' 

    def __init__(self, definition):
        '''
        Creates equation if `definition` is of the right form.

        **Parameters**

        definition : str
            Equation definition of the form 'y = expr : dtype'
            expr must be a valid python expression.

        '''
        Definition.__init__(self,definition)
        self._parse(definition)


    def _parse(self, definition):
        '''
        Parse definition and check it is an equation.

        **Parameters**

        definition : str
            Equation definition of the form 'y = expr : dtype'
            expr must be a valid python expression.
        '''
        self._definition = definition
        definition = str(definition.replace(' ',''))

        # Check if equation is of the form: y = f(...) : dtype
        p = re.compile(
            r'''(?P<y>\w+) = (?P<f>[^:]+) (:(?P<dtype>\w+))?''', re.VERBOSE)
        result = p.match(definition)
        if result:
            y = result.group('y').strip()
            f = result.group('f').strip()
            dtype = (result.group('dtype') or 'float').strip()
            self._varname = y
            self._lhs = y
            self._rhs = f
            self.__f__ = eval('lambda: %s' % f)

            # This line get all strings from function code definition...
            self._variables = list(self.__f__.func_code.co_names)
            #  ... and we need to sort out actual variables from function names
            namespace = {}
            numpy_ns = {}
            for key in dir(np):
                numpy_ns[key] = getattr(np,key)
            for i in range(0,len(inspect.stack())):
                frame = inspect.stack()[i][0]
                for name in self.__f__.func_code.co_names:
                    if (name in self._variables) and (name != self._varname):
                        if name in numpy_ns.keys() and callable(eval(name, numpy_ns)):
                            namespace[name] = numpy_ns[name]
                            self._variables.remove(name)
                        if name in frame.f_locals.keys() and callable(eval(name, frame.f_locals)):
                            namespace[name] = frame.f_locals[name]
                            self._variables.remove(name)
            args = self.__f__.func_code.co_names
            if len(args):
                args = ' = 0, '.join(self._variables)+ ' = 0'
            else:
                args = ''
            self.__f__ = eval('lambda %s: %s' % (args,f), namespace)
            self._dtype = dtype
        else:
            raise EquationError, 'Definition is not an equation'


    def __call__(self, *args, **kwargs):
        '''
        Evaluate equation
        
        **Parameters**

        args : list
            Equation constants (respecting definition order)
        kwargs : dict
            Equation constants
            
        **Examples**

        >>> eq = Equation('y = a+b*x')
        >>> y = eq.evaluate(1,2,3)             # a=1, b=2, x=3
        >>> y = eq.evaluate(x=1, b=2, a=3)     # a=3, b=2, x=1
        '''

        return self.__f__(*args, **kwargs)


    def evaluate(self, *args, **kwargs):
        '''
        Evaluate equation
        
        **Parameters**

        args : list
            Equation constants (respecting definition order)
        kwargs : dict
            Equation constants
            
        **Examples**

        >>> eq = Equation('y = a+b*x')
        >>> y = eq.evaluate(1,2,3)             # a=1, b=2, x=3
        >>> y = eq.evaluate(x=1, b=2, a=3)     # a=3, b=2, x=1
        '''
        return self.__f__(*args, **kwargs)



    def _get_variables(self):
        '''Get equation variables'''
        return self._variables
    variables = property(_get_variables,
                         doc='''Equation variable names''')



if __name__ == '__main__':
    y, t,dt = 0.0, 1.0,  0.01
    eq = Equation(u'y = (alpha + beta*y)*dt')
    for i in range(int(t/dt)):
        y += eq.evaluate(alpha=1, beta=1, y=y, dt=dt)
    print 'Equation evaluation:', y

