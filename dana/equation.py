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
"""
import re
import inspect
import compiler
from definition import Definition, Visitor


class EquationError(Exception):
    pass

class Equation(Definition):
    """ Equation of type: 'y = expr : dtype'

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
    """

    def __init__(self, definition, constants=None, variables=None):
        """
        Creates equation.

        :param string definition:
            Equation definition of the form:
        :param list constants:
            Name of variables that must be considered constant
        """
        if not variables: variables = []
        if not constants: constants = {}
        Definition.__init__(self,definition)
        self._in_out = None
        self._out = None
        self.setup()


    def setup(self, constants=None):
        """
        Parse definition and check it is an equation.

        **Parameters**

        definition : str
            Equation definition of the form 'y = expr : dtype'
            expr must be a valid python expression.
        """
        if not constants: constants = {}

        # Check if equation is of the form: y = f(...) : dtype
        p = re.compile(
            r'''(?P<y>\w+) = (?P<f>[^:]+) (:(?P<dtype>\w+))?''', re.VERBOSE)
        result = p.match(self._definition)
        if result:
            y = result.group('y').strip()
            f = result.group('f').strip()
            dtype = (result.group('dtype') or 'float').strip()
            self._lhs = y
            self._varname = y
            self._rhs = f
            self._dtype = dtype
            visitor = Visitor()
            compiler.walk(compiler.parse(f), visitor)
            variables = visitor._vars

            # Make sure to get function form highest stack frame
            # since the function name can be also defined locally
            ns = {}
            for name in visitor._funcs:
                for i in range(1,len(inspect.stack())):
                    frame = inspect.stack()[i][0]
                    name = name.split('.')[0]
                    if name in frame.f_locals.keys() and name not in ns.keys():
                        ns[name] = frame.f_locals[name]
                        break
            ns.update(constants)

            variables = list(set(variables) - set(constants.keys()))
            self._variables = variables

            if len(variables):
                args = ' = 0, '.join(variables)+ ' = 0'
            else:
                args = ''
            self.__f__ = eval('lambda %s: %s' % (args,f),ns)
        else:
            raise EquationError, 'Definition is not an equation'


    def __call__(self, *args, **kwargs):
        """
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
        """
        
        return self.__f__(*args, **kwargs)


    def evaluate(self, *args, **kwargs):
        """
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
        """

        return self.__f__(*args, **kwargs)



    def _get_variables(self):
        """Get equation variables"""
        return self._variables
    variables = property(_get_variables,
                         doc='''Equation variable names''')



if __name__ == '__main__':
    y, t,dt = 0.0, 1.0,  0.01
    eq = Equation(u'y = (alpha + beta*y)*dt')
    for i in range(int(t/dt)):
        y += eq.evaluate(alpha=1, beta=1, y=y, dt=dt)
    print 'Equation evaluation:', y

