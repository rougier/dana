#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
"""
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
"""
import re
import inspect
import compiler
import numpy as np
from definition import Definition, Visitor


class DifferentialEquationError(Exception):
    pass

class DifferentialEquation(Definition):
    """ DifferentialEquation of type: 'dy/dt = expr'

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
    """

    def __init__(self, definition, constants=None):
        """
        Creates differential equation.

        :param string definition:
            Equation definition

        :param list constants:
            Name of variables that must be considered constant
        """
        if not constants: constants = {}
        Definition.__init__(self, definition)
        self._in_out = None
        self._out = None
        self.setup()
        self.__method__ = self._forward_euler

    def __repr__(self):
        """ x.__repr__() <==> repr(x) """

        classname = self.__class__.__name__
        return "%s('d%s/dt = %s : %s')" % (classname, self._lhs, self._rhs, self._dtype)

    def setup(self, constants=None):
        """
        Parse definition and check if it is an equation.

        :param string definition:
            Equation definition of the form:

             dy/dt = expr : dtype or dy/dt = A+(B)*y : dtype

            expr, A and B must be valid python expressions.

        :param list constants:
            List of variable names that must be considered constants.
        """
        if not constants: constants = {}

        # First, we check if equation is of the form: dy/dt = A + (B)*y [: dtype]
        # -----------------------------------------------------------------------
        p = re.compile(
            r'''d(?P<y>\w+)/dt = (?P<A>.+?)? (?P<sign>\+|-)? \((?P<B>.*)\)\*(?P=y)
                           (: (?P<dtype>.+))?''',re.VERBOSE)
        result = p.match(self._definition)
        if result:
            y = result.group('y').strip()
            A = (result.group('A') or '0').strip()
            sign = (result.group('sign') or '+').strip()
            B = result.group('B').strip()
            if A == '-' or A == '+':
                sign, A = A, '0'
            if sign == '+':
                B = '-' + B
            dtype = (result.group('dtype') or 'float')


            # Check that var is not in A nor in B
            if (y not in compile(A,'<string>', 'eval').co_names
                and y not in compile(B,'<string>', 'eval').co_names):
                self._lhs = y
                self._varname = y
                self._rhs = '%s%s(%s)*%s' % (A,sign,B,y)
                self._dtype = dtype
                visitor = Visitor()
                compiler.walk(compiler.parse('%s+%s' % (A,B)), visitor)
                variables = visitor._vars

                ns = {}
                for name in visitor._funcs:
                    for i in range(1,len(inspect.stack())):
                        frame = inspect.stack()[i][0]
                        name = name.split('.')[0]
                        if name in frame.f_locals.keys() and name not in ns.keys():
                            ns[name] = frame.f_locals[name]
                            break
                ns.update(constants)
                if y in variables:
                    variables.remove(y)
                variables = list(set(variables) - set(constants.keys()))
                self._variables = variables
                if len(variables):
                    args = ' = 0, '.join(variables)+ ' = 0'
                else:
                    args = ''
                self.__f__ = eval('lambda %s,%s: %s-%s*%s' % (y,args,A,B,y),ns)
                self.__A__ = eval('lambda %s: %s' % (args,A),ns)
                self.__B__ = eval('lambda %s: %s' % (args,B),ns)
                self._A_string = A
                self._B_string = B
                self._dtype = dtype
                return

        # Second, we check if equation is of the form: dy/dt = f(y)
        # ---------------------------------------------------------
        p = re.compile(
            r"d(?P<y>\w+)/dt = (?P<f>[^:]+) (: (?P<dtype>\w+))?", re.VERBOSE)
        result = p.match(self._definition)
        if result:
            y = result.group('y').strip()
            f = result.group('f').strip()
            dtype = (result.group('dtype') or 'float').strip()
            self._lhs = y
            self._rhs = f
            self._dtype = dtype
            self._varname = y
            visitor = Visitor()
            compiler.walk(compiler.parse(f), visitor)
            variables = visitor._vars

            ns = {}
            for name in visitor._funcs:
                for i in range(1,len(inspect.stack())):
                    frame = inspect.stack()[i][0]
                    name = name.split('.')[0]
                    if name in frame.f_locals.keys() and name not in ns.keys():
                        ns[name] = frame.f_locals[name]
                        break
            ns.update(constants)
            if y in variables:
                variables.remove(y)
            variables = list(set(variables) - set(constants.keys()))
            self._variables = variables
            if len(variables):
                args = ' = 0, '.join(variables)+ ' = 0'
            else:
                args = ''
            self.__f__ = eval('lambda %s,%s: %s' % (y,args,f), ns)
            self._f_string = 'lambda %s,%s: %s' % (y,args,f)
            self.__A__ = None
            self.__B__ = None
            self._A_string = ""
            self._B_string = ""
            return

        # Last case, it is not a differential equation
        # --------------------------------------------
        raise DifferentialEquationError, \
            'Equation is not a first order differential equation'

    def _forward_euler(self, __x__, dt, *args, **kwargs):
        """
        Forward euler method evaluation method

        **Notes:**

        See ``evaluate`` method for parameters
        """
        dx = self.__f__(__x__, *args, **kwargs)*dt
        if self._out is not None:
            np.add(__x__, dx, self._out)
            return self._out
        elif self._in_out is not None:
            __x__ += dx
            return __x__
        return __x__ + dx

    def _runge_kutta_2(self, __x__, dt, *args, **kwargs):
        """
        Runge Kutta 2nd order evaluation method.

        **Notes**

        See ``evaluate`` method for parameters.
        """
        __hdt = 0.5*dt
        __k1 = self.__f__(__x__, *args, **kwargs)
        __k2 = self.__f__(__x__ + dt*__x__, *args, **kwargs)
        dx = 0.5*dt*(__k1 + __k2)
        if self._out is not None:
            np.add(__x__, dx, self._out)
            return self._out
        elif self._in_out is not None:
            __x__ += dx
            return __x__
        return __x__ + dx

    def _runge_kutta_4(self, __x__, dt, *args, **kwargs):
        """
        Runge Kutta 4th order evaluation method.

        **Notes**

        See ``evaluate`` method for parameters.
        """
        __hdt = 0.5*dt
        __k1 = self.__f__(__x__, *args, **kwargs)
        __k2 = self.__f__(__x__ + __k1 * __hdt, *args, **kwargs)
        __k3 = self.__f__(__x__ + __k2 * __hdt, *args, **kwargs)
        __k4 = self.__f__(__x__ + __k3 * dt, *args, **kwargs)
        dx  = (__k1+__k4)*(dt/6.0)+(__k2+__k3)*(dt/3.0)
        if self._out is not None:
            np.add(__x__, dx, self._out)
            return self._out
        elif self._in_out is not None:
            __x__ += dx
            return __x__
        return __x__ + dx

    def _exponential_euler(self, __x__, dt, *args, **kwargs):
        """
        Exponential Euler evaluation method.

        **Notes**

        See ``evaluate`` method for parameters.
        Only available for equation of the form dy/dt = A + B*y
        """
        A = float(self.__A__(*args, **kwargs))
        B = float(self.__B__(*args, **kwargs))
        AB = A
        AB /= B
        E = np.exp(-B*dt)

        if self._out is not None:
            np.mul(__x__,E,self._out)
            __x__ = self._out
        elif self._in_out is not None:
            __x__ *=  E
        else:
            __x__ =  __x__*E
        __x__ +=  AB
        AB *= E
        __x__ -= AB
        return __x__

    def select(self, method = 'Forward Euler'):
        """
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
        """

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
        """
        Evaluate __x__ at time t+dt.

        :param __x__:
             Variable that need to be evaluated
        :param dt:
            Elementaty time step
        :param list args:
            Optional arguments
        :param dict kwargs:
            Optional keyword arguments

        **Notes**

        All equation constants should be given in args (respecting order from
        definition) or in kwargs (any order).

        **Examples**

        >>> y = 0
        >>> eq = DifferentialEquation('dy/dt = a+b*y')
        >>> y = eq.evaluate(y, 0.01, 1, 2)     # a=1, b=2
        >>> y = eq.evaluate(y, 0.01, b=1, a=2) # a=2, b=1
        """
        return self.__method__ (__x__, dt, *args, **kwargs)

    def evaluate(self, __x__, dt, *args, **kwargs):
        """
        Evaluate __x__(t+dt)

        :param __x__:
             Variable that need to be evaluated
        :param dt:
            Elementaty time step

        **Notes**

        All equation constants should be given in args (respecting order from
        definition) or in kwargs (any order).

        **Examples**

        >>> y = 0
        >>> eq = DifferentialEquation('dy/dt = a+b*y')
        >>> y = eq.evaluate(y, 0.01, 1, 2)     # a=1, b=2
        >>> y = eq.evaluate(y, 0.01, b=1, a=2) # a=2, b=1
        """
        return self.__method__ (__x__, dt, *args, **kwargs)



# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np
    
    t, dt = 1.0, 0.001
    eq = DifferentialEquation('dz/dt=a + (b)*z''')

    eq.select('Forward Euler')
    y = 1.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=0, b=1)
    print 'Forward Euler:      ', y

    eq.select('Runge Kutta 2')
    y = 1.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=0, b=1)
    print 'Runge Kutta 2:      ', y

    eq.select('Runge Kutta 4')
    y = 1.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=0, b=1)
    print 'Runge Kutta 4:      ', y
    
    eq.select('Exponential Euler')
    y = 1.0
    for i in range(int(t/dt)):
        y = eq(y, dt, a=0, b=1)
    print 'Exponential Euler:  ',y
