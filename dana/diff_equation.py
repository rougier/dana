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
DifferentialEquation class.

The DifferentialEquation class allows to manipulate equations of the type:

   dy/dt = expr : dtype

where `y` is the equation variable, `expr` is a valid python expression and
`dtype` describes the type of the `y` variable. When an equation is created,
the given definition is parsed to check if is of the right form. If this is not
the case, a `DifferentialEquationError` is raised.

**Examples**

>>> y = 0
>>> eq = DifferentialEquation('dy/dt = a+b*y')
>>> y = eq.evaluate(y, 0.01, 1, 2)     # a=1, b=2
>>> y = eq.evaluate(y, 0.01, b=1, a=2) # a=2, b=1
'''

import re
import numpy as np
import inspect
from definition import Definition


class DifferentialEquationError(Exception):
    pass

class DifferentialEquation(Definition):
    ''' DifferentialEquation of type: 'dy/dt = expr' 

    The DifferentialEquation class allows to manipulate equations of the type:

        dy/dt = expr : dtype

    where `y` is the equation variable, `expr` is a valid python expression and
    `dtype` describes the type of the `y` variable. When a differential
    equation is created, the given definition is parsed to check if is of the
    right form. If this is not the case, a `DifferentialEquationError` is
    raised.

    **Examples**

    >>> eq = DifferentialEquation('dy/dt = a+b*x')
    >>> y = eq.evaluate(0.0, dt=0.1, a=1, b=2)
    0.1
    '''

    def __init__(self, definition):
        '''
        Creates equation if `definition` is of the right form.

        **Parameters**

        definition : str
            Equation definition of the form 'dy/dt = expr : dtype'
            expr must be a valid python expression.
        '''
        Definition.__init__(self,definition)
        self._parse(definition)
        self.__method__ = self._forward_euler


    def __repr__(self):
        ''' x.__repr__() <==> repr(x) '''

        classname = self.__class__.__name__
        return "%s('d%s/dt = %s : %s')" % (classname, self._lhs, self._rhs, self._dtype)


    def _parse(self, definition):
        '''
        Parse definition and check if it is an equation.

        **Parameters**

        definition : str
            Equation definition of the form 'y = expr : dtype'
            expr must be a valid python expression.
        '''
        self._definition = definition
        definition = str(definition.replace(' ',''))

        # First, we check if equation is of the form:
        #   dy/dt = A + (B)*y [: dtype]
        p = re.compile(r'''d(?P<y>\w+)/dt =
                           (?P<A>.+?)? (?P<sign>\+|-)? \((?P<B>.*)\)\*(?P=y)
                           (: (?P<dtype>.+))?''',re.VERBOSE)
        result = p.match(definition)
        if result:
            y = result.group('y')
            A  = result.group('A') or '0'
            sign = result.group('sign') or '+'
            B = result.group('B')
            if A == '-' or A == '+':
                sign = A
                A = '0'
            dtype = (result.group('dtype') or 'float')

            # Check that var is not in A nor B
            if (y not in compile(A,'<string>', 'eval').co_names
                and y not in compile(B,'<string>', 'eval').co_names):
                self._varname = y
                self._lhs = 'd%s/dt' % y
                self.__f__ = eval('lambda %s: %s+%s*%s' % (y, A, B, y))

                # This line get all strings from function code definition...
                self._variables = list(self.__f__.func_code.co_names)
                #  ... and we need to sort out actual variables from function names
                numpy_ns = {}
                for key in dir(np):
                    numpy_ns[key] = getattr(np,key)
                for i in range(0,len(inspect.stack())):
                    frame = inspect.stack()[i][0]
                    for name in self.__f__.func_code.co_names:
                        if ((name in self._variables) and (name != self._varname) and
                            (name in numpy_ns.keys() or name in frame.f_globals.keys()) and
                            callable(eval(name, numpy_ns, frame.f_globals))):
                            self._variables.remove(name)

                args = self.__f__.func_code.co_names
                if len(args):
                    args = ' = 0, '.join(self._variables)+ ' = 0'
                else:
                    args = ''
                self.__f__ = eval('lambda %s,%s: %s+%s*%s' % (y,args,A,B,y), numpy_ns)
                self._rhs = '%s+%s*%s' % (A,B,y)
                self.__A__ = eval('lambda : %s+%s' % (A,B), numpy_ns)
                args = self.__A__.func_code.co_names
                if len(args):
                    args = ' = 0, '.join(args)+ ' = 0'
                else:
                    args = ''
                self.__A__ = eval('lambda %s: %s' % (args,A), numpy_ns)
                self._A_string = A
                self.__B__ = eval('lambda %s: %s' % (args,B), numpy_ns)
                self._B_string = B
                self._dtype = dtype
                #self._variables = self._f.func_code.co_names
                return

        # Second, we check if equation is of the form: dy/dt = f(y)
        p = re.compile(r'''d(?P<y>\w+)/dt =
                           (?P<f>[^:]+)
                           (: (?P<dtype>\w+))?''', re.VERBOSE)
        result = p.match(definition)
        if result:
            y = result.group('y').strip()
            f = result.group('f').strip()
            dtype = (result.group('dtype') or 'float').strip()
            self._varname = y
            self._lhs = y
            self._rhs = f
            self.__f__ = eval('lambda %s: %s' % (y,f))
            # This line get all strings from function code definition...
            self._variables = list(self.__f__.func_code.co_names)
            #  ... and we need to sort out actual variables from function names
            numpy_ns = {}
            for key in dir(np):
                numpy_ns[key] = getattr(np,key)
            for i in range(0,len(inspect.stack())):
                frame = inspect.stack()[i][0]
                for name in self.__f__.func_code.co_names:
                    if ((name in self._variables) and (name != self._varname) and
                        (name in numpy_ns.keys() or name in frame.f_globals.keys()) and
                        callable(eval(name, numpy_ns, frame.f_globals))):
                        self._variables.remove(name)
            args = self.__f__.func_code.co_names
            if len(args):
                args = ' = 0, '.join(self._variables)+ ' = 0'
            else:
                args = ''
            self.__f__ = eval('lambda %s,%s: %s' % (y,args,f), numpy_ns)
            self.__A__ = None
            self._A_string = ''
            self.__B__ = None
            self._B_string = ''
            self._dtype = dtype
            return

        # Last case, it is not a differential equation
        raise DifferentialEquationError, \
            'Equation is not a first order differential equation'


    def _forward_euler(self, __x__, dt, *args, **kwargs):
        '''
        Forward euler method evaluation method

        **Notes:**

        See ``evaluate`` method for parameters
        '''
        __x__ += self.__f__(__x__, *args, **kwargs)*dt
        return __x__


    def _runge_kutta_2(self, __x__, dt, *args, **kwargs):
        '''
        Runge Kutta 2nd order evaluation method.

        **Notes**

        See ``evaluate`` method for parameters.
        '''
        __hdt = 0.5*dt
        __k1 = self.__f__(__x__, *args, **kwargs)
        __k2 = self.__f__(__x__ + dt*__x__, *args, **kwargs)
        __x__ += 0.5*dt*(__k1 + __k2)
        return __x__


    def _runge_kutta_4(self, __x__, dt, *args, **kwargs):
        '''
        Runge Kutta 4th order evaluation method.

        **Notes**

        See ``evaluate`` method for parameters.
        '''
        __hdt = 0.5*dt
        __k1 = self.__f__(__x__, *args, **kwargs)
        __k2 = self.__f__(__x__ + __k1 * __hdt, *args, **kwargs)
        __k3 = self.__f__(__x__ + __k2 * __hdt, *args, **kwargs)
        __k4 = self.__f__(__x__ + __k3 * dt, *args, **kwargs)
        __x__ += (__k1+__k4)*(dt/6.0)+(__k2+__k3)*(dt/3.0)
        return __x__


    def _exponential_euler(self, __x__, dt, *args, **kwargs):
        '''
        Exponential Euler evaluation method.

        **Notes**

        See ``evaluate`` method for parameters.
        Only available for equation of the form dy/dt = A + B*y
        '''
        A = self.__A__(*args, **kwargs)
        B = self.__B__(*args, **kwargs)
        AB = A/B
        E = np.exp(B*dt)
        __x__ *=  E
        __x__ -=  AB
        AB *= E          
        __x__ += AB
        return __x__


    def select(self, method = 'Forward Euler'):
        '''
        Select evaluation method.

        **Parameters**

        method : 
            * 'Forward Euler'
            * 'Runge Kutta 2'
            * 'Runge Kutta 4'
            * 'Exponential Euler'

        **Notes**

        Exponential Euler method is only available for equation of type:
        dy/dt = A+(B)*y with A,B being valid python expression.
        '''

        if method == 'Forward Euler':
            self.__method__ = self._forward_euler
        elif method == 'Runge Kutta 2':
            self.__method__ = self._runge_kutta_2
        elif method == 'Runge Kutta 4':
            self.__method__ = self._runge_kutta_4
        elif method == 'Exponential Euler':
            if self.__A__ is not None and self.__B__ is not None:
                self.__method__ = self._exponential_euler
            else:
                raise DifferentialEquationError, \
                    '''Equation '%s' is not of type dy/dt = A + (B)*y''' % repr(self)
        else:
            raise DifferentialEquationError, \
                'Unknown evaluation method (%s)' % method

    def __call__(self, __x__, dt, *args, **kwargs):
        '''
        Evaluate __x__(t+dt)
        
        **Parameters**

        __x__ : float
            Variable that need to be evaluated

        dt : float
            Elementaty time step

        **Notes**

        All equation constants should be given in args (respecting order from
        definition) or in kwargs (any order).

        **Examples**

        >>> y = 0
        >>> eq = DifferentialEquation('dy/dt = a+b*y')
        >>> y = eq.evaluate(y, 0.01, 1, 2)     # a=1, b=2
        >>> y = eq.evaluate(y, 0.01, b=1, a=2) # a=2, b=1
        '''
        return self.__method__ (__x__, dt, *args, **kwargs)


    def evaluate(self, __x__, dt, *args, **kwargs):
        '''
        Evaluate __x__(t+dt)
        
        **Parameters**

        __x__ : float
            Variable that need to be evaluated

        dt : float
            Elementaty time step

        **Notes**

        All equation constants should be given in args (respecting order from
        definition) or in kwargs (any order).

        **Examples**

        >>> y = 0
        >>> eq = DifferentialEquation('dy/dt = a+b*y')
        >>> y = eq.evaluate(y, 0.01, 1, 2)     # a=1, b=2
        >>> y = eq.evaluate(y, 0.01, b=1, a=2) # a=2, b=1
        '''
        return self.__method__ (__x__, dt, *args, **kwargs)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    t, dt = 10.0, 0.01
    eq = DifferentialEquation('dy/dt = a + (b)*y''')
    #eq = DifferentialEquation('dy/dt = a + b*y''')

    eq.select('Forward Euler')
    y = 0.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=1, b=1)
    print 'Forward Euler:      ', y

    eq.select('Runge Kutta 2')
    y = 0.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=1, b=1)
    print 'Runge Kutta 2:      ', y

    eq.select('Runge Kutta 4')
    y = 0.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=1, b=1)
    print 'Runge Kutta 4:      ', y
    
    eq.select('Exponential Euler')
    y = 0.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=1, b=1)
    print 'Exponential Euler:  ',y
