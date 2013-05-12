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
import inspect
from diff_equation import DifferentialEquation
import numpy as np


class ConnectionError(Exception):
    """ Connection Error """
    pass


class Connection(object):
    """
    A connection describes a flow of information between two groups (that can
    possibly be the same). It is characterized by a source group, a target
    group and a model describing both how to compute connection output, what
    state variable of the destination group holds the result and how connection
    evolves through time (if specified).

    Left-hand side of the equation is the name of the state variable while
    right-hand side is the expression to be evaluated to get the actual
    output. Any state variable within the right-hand side of the equation
    refers to the source group if it exists.

    **Example:**

      >>> src = numpy.ones((3,3))
      >>> tgt = numpy.ones((3,3), dtype=[('U',float),('V',float)])
      >>> K = np.ones((src.size,))
      >>> C = Connection(src, dst('U'), 'U = V*K')
      >>> C.setup(), C.propagate()

    In the above example, a connection has been created between ``src`` and
    ``tgt``. The state variable holding the result in ``tgt`` is named ``U``
    and the output of the connection is computed by mutliplying src by ``K``.
    """

    def __init__(self, source, target, toric=False):

        """
        Constructs a new connection between a source and a target using
        specified connection model.

        **Parameters**

        source : Group
            Source group
        target : Group
            Target group
        """

        self._weights = None
        self._equation = None
        self._toric = toric

        # Get actual source
        names = source.dtype.names
        if names is None:
            self._actual_source = source
            self._source_name = ''
        else:
            self._actual_source = source[names[0]]
            self._source_name = names[0]

        # Get actual target
        names = target.dtype.names
        if names is None:
            self._actual_target = target
            self._target_name = ''
        else:
            self._actual_target = target[names[0]]
            self._target_name = names[0]

        # Get source base group
        if source.base is None:
            self._source = source
        else:
            self._source = source.base

        # Get target base group
        if target.base is None:
            self._target = target
        else:
            self._target = target.base

        # Append this connection to target connections
        if hasattr(self._target, '_connections'):
            self._target._connections.append(self)

    def setup_weights(self, weights):
        """ Setup weights if necessary """
        pass

    def setup_equation(self, equation):
        """ Setup weights update equation """

        if not equation:
            self._equation = None
            self._kwargs = None
            return
        equation = equation.replace("pre.",  "pre_")
        equation = equation.replace("post.", "post_")
        #
        # TODO : Replace any dotted variable with actual value in namespace
        #
        eq = DifferentialEquation(equation)
        kwargs = {}
        src = self.source
        tgt = self.target
        for variable in eq._variables:
            if variable == "pre":
                kwargs[variable] = self._actual_source.reshape((1,src.size))
            elif variable == "post":
                kwargs[variable] = self._actual_target.reshape((tgt.size,1))
            elif variable.startswith("pre_"):
                kwargs[variable] = src[variable[4:]].reshape((1,src.size))
            elif variable.startswith("post_"):
                kwargs[variable] = tgt[variable[5:]].reshape((tgt.size,1))
#            elif variable in tgt.dtype.names:
#               kwargs[variable] = tgt[variable].reshape((tgt.size,1))
            else:
                for i in range(1,len(inspect.stack())):
                    frame = inspect.stack()[i][0]
                    if variable in frame.f_globals.keys() and variable not in kwargs:                 
                        kwargs[variable] = frame.f_globals[variable]
        self._equation = eq
        self._kwargs = kwargs

    def propagate(self):
        """ Propagate activity from source to target """

        if self._source_name:
            self._actual_source = self._source._data[self._source_name]
        if self._target_name:
            self._actual_target = self._target._data[self._target_name]
        self._actual_target += self.output()


    def evaluate(self, dt=0.01):
        """ Update weights relative to connection equation """
        if not self._equation:
            return
        pre, post = self._source, self._target
        for arg in self._kwargs.keys():
            if arg.startswith("pre_"):
                self._kwargs[arg] = pre[arg[4:]].reshape((1,pre.size))
            elif arg.startswith("post_"):
                self._kwargs[arg] = post[arg[5:]].reshape((post.size,1))
        self._equation._in_out = self._weights
        self._equation.evaluate(self._weights, dt, **self._kwargs)

    def output(self):
        """ Return output of connection """
        raise NotImplementedError

    def __getitem__(self, key):
        """ Return connection from """
        raise NotImplementedError

    source = property(lambda self: self._source,
        doc='''Source of the connection.''')

    source_name = property(lambda self: self._source_name,
        doc='''Name of the source value of the connection.''')

    target = property(lambda self: self._target,
        doc='''Target of the connection.''')

    target_name = property(lambda self: self._target_name,
        doc='''Name of the target value of the connection.''')

    weights = property(lambda self: self._weights,
        doc='''Weights matrix.''')

    equation = property(lambda self: self._equation,
        doc='''Differential equation for weights update''')


# ---------------------------------------------------------------- __main__ ---
if __name__ == '__main__':
    from group import Group

    src = Group((3,3), 'V,U')
    tgt = Group((3,3), 'V,U')
    C = Connection(src('V'), tgt('U'))
    print C.source == src
    print C.target == tgt

