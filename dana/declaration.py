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
import re
from dana.definition import Definition

class DeclarationError(Exception):
    pass

class Declaration(Definition):
    """ Declaration of type: 'y : dtype' """
  
    def __init__(self, definition, constants=None):
        """ Builds a new Declaration of type: 'y : dtype' """
        if not constants: constants = {}
        Definition.__init__(self, definition, constants)
        self.setup()


    def setup(self, constants=None):
        """
        Parse definition and check it is a declaration.

        **Parameters**

        definition : str
            Equation definition of the form 'y : dtype'
        """
        if not constants: constants = {}

        p = re.compile(r'''(?P<y>\w+) (:(?P<dtype>\w+))?''', re.VERBOSE)
        result = p.match(self._definition)
        if result:
            self._varname = result.group('y')
            self._lhs = self._varname
            self._rhs = ''
            self._dtype = result.group('dtype') or 'float'
        else:
            raise DeclarationError, 'Definition is not a declaration'


    def __call__(self):
        """
        Evaluate declaration (return dtype)
        """

        return eval(self.dtype)


    def evaluate(self):
        """
        Evaluate declaration (return dtype)
        """

        return eval(self.dtype)


    def __repr__(self):
        """ x.__repr__() <==> repr(x) """

        classname = self.__class__.__name__
        return "%s('%s : %s')" % (classname, self._varname, self._dtype)


if __name__ == '__main__':
    eq = Declaration('x : float')
