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
Model class
"""
import re
from equation import Equation, EquationError
from definition import Definition, DefinitionError
from declaration import Declaration, DeclarationError
from diff_equation import DifferentialEquation, DifferentialEquationError


# ------------------------------------------------------------------- parse ---
def parse(definition):
    """ Parse a definition and return the corresponding object.

    **Parameters**

    definition : str
       String describing equation, differential equation, declaration or alias
    """

    try:
        return DifferentialEquation(definition)
    except DifferentialEquationError:
        pass
    try:
        return Equation(definition)
    except EquationError:
        pass
    try:
        return Declaration(definition)
    except DeclarationError:
        pass
    raise ValueError, \
        'Definition has not been recognized ("%s")' % definition

# --------------------------------------------------------------- ModelError ---
class ModelError(Exception):
    """ Model Exception """
    pass

# -------------------------------------------------------------------- Model ---
class Model(object):
    """
    A model is a set of equations that are supposed to be evaluated
    together. Those equations can be:

    * :class:`DifferentialEquation` of the form ``dy/dt = expr : dtype``
    * :class:`Equation` of the form ``y = expr : dtype``
    * :class:`Declaration` of the form ``y : dtype``
    
    where ``expr`` is a valid python expression.

    **Examples**

    >>> model = Model('''dx/dt = 1.0 : float  # differential equation
                            y  = 1.0 : float  # equation''')
    """

    def __init__(self, definition):
        """
        A model is a set of value that can evolve through time and learning and is
        described by a set of equations that can be a

        * differential equation of the form ``dy/dt = expr : dtype``
        * equation of the form ``y = expr : dtype``
    
        where ``expr`` is a valid python expression.

        **Examples**

        >>> model = Model('''dx/dt = 1.0 : float  # differential equation
                           y  = 1.0    : float  # equation
                           z           : float  # declaration''')
        """
        self._diff_equations = []
        self._equations = []
        self._declarations = []
        self._all = []
        self._variables = []
        self._definition = definition
        self.setup()


    def setup(self):
        """ Parse definition and instantiate corresponding objects """

        self._diff_equations = []
        self._equations = []
        self._declarations = []
        self._variables = []
        self._all = []
        definition = re.sub('\\\s*?\n', ' ', self._definition)
        for line in re.split('[\n;]', definition):
            line = line.strip()
            if len(line) and line[0] != '-':
                equation = parse(line)
                for other in self._all:
                    if equation.varname == other.varname:
                            # and type(equation) == type(other):
                        raise ModelError, \
                            '%s is already defined (%s)' % (equation.varname,
                                                            other.definition)
                self._all.append(equation)
                if isinstance(equation, Declaration):
                    self._declarations.append(equation)
                    varname = equation.varname
                    if varname not in self._variables:
                        self._variables.append(varname)
                elif isinstance(equation, Equation):
                    self._equations.append(equation)
                    varname = equation.varname
                    if varname not in self._variables:
                        self._variables.append(varname)
                elif isinstance(equation, DifferentialEquation):
                    self._diff_equations.append(equation)
                    varname = equation.varname
                    if varname not in self._variables:
                        self._variables.append(varname)
                    
               
        # Check for circular dependencies in equations and order equations
        # relatively to their inter-dependencies
        if self._equations:
            #v = [eq._varname for eq in self._equations]
            variables = [eq.varname for eq in self._equations]
            ordered = []
            while variables:
                dependencies = {}
                for i in range(len(variables)):
                    var = variables[i]
                    equation = self[var]
                    dependencies[var] = [v for v in equation._variables
                                         if v in variables and v != var]
                if [] not in dependencies.values():
                    raise ModelError, \
                        'Model possesses circular dependencies in equations'

                # Be careful to keep equations original order if possible
                for equation in self._equations:
                    if equation.varname in dependencies.keys():
                        key = equation.varname
                        if not dependencies[key]:
                            variables.remove(key)
                            ordered.append(self[key])
            self._equations = ordered


    def run(self, namespace=None, dt=0.001):
        """ Run the model model within the given namespace

        **Parameters**

        namespace : dict
            Dictionnary of necessary variables to run the model

        dt : float
            Elementaty time step

        **Examples**

        >>> model = Model('dx/dt = 1.0; y = 1.0')
        >>> model.run({'x':0}, dt=0.01)
        {'y': 1.0, 'x': 0.01}
        """

        if namespace is None:
            namespace = {}
        for eq in self._diff_equations:
            args = [namespace[eq._varname],dt]+ \
                [namespace[var] for var in eq._variables]
            namespace[eq._varname] = eq.evaluate(*args)
        for eq in self._equations:
            args = [namespace[var] for var in eq._variables]
            namespace[eq._varname] = eq.evaluate(*args)
        return namespace

    def __getattr__(self, key):
        """ x.__getattribute__(key) <==> x.name """

        try:
            return  self.__getitem__(key)
        except ModelError:
            return object.__getattribute__(self, key)


    def __getitem__(self, key):
        """ x.__getitem__(y) <==> x[y] """

        for equation in self._all:
            if equation.varname == key:
                return equation
        raise ModelError, 'There is no definition for %s' % key

    def __iter__(self):
        for eq in self._all:
            yield eq


    def __repr__(self):
        """ x.__repr__() <==> repr(x) """
        
        string = ''
        for equation in self._all:
            string += repr(equation)+'\n'
        return string[:-1]

    def _get_declarations(self):
        return self._declarations
    declarations = property(_get_declarations,
                         doc = ''' Model declarations ''')

    def _get_equations(self):
        return self._equations
    equations = property(_get_equations,
                         doc = ''' Model equations ''')

    def _get_diff_equations(self):
        return self._diff_equations
    diff_equations = property(_get_diff_equations,
                         doc = ''' Model differental equations ''')

    def _get_connections(self):
        return self._connections
    connections = property(_get_connections,
                           doc = ''' Model connections ''')

    def _get_variables(self):
        """Get model variables"""
        return self._variables
    variables = property(_get_variables,
                         doc='''Model variable names''')


# ---------------------------------------------------------------- __main__ ---
if __name__ == '__main__':
    model = Model('''dx/dt = 1.0 : float  # differential equation
                     y     = 1.0 : float  # equation
                     z           : float  # declaration''')
    print model
