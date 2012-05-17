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
import numpy as np
import scipy.sparse as sp

def np_almost_equal(A, B, epsilon=1e-10):
    """ Assert two arrays are almost equal, even with NaN in them """
    if sp.issparse(A):
        A = A.todense()
    if sp.issparse(B):
        B = B.todense()
    A_nan = np.isnan(A)
    B_nan = np.isnan(B)
    A_num = np.nan_to_num(A)
    B_num = np.nan_to_num(B)
    return np.all(A_nan==B_nan) and (np.abs(A_num-B_num)).sum() <= epsilon

def np_equal(A, B):
    """ Assert two arrays are equal, even with NaN in them """
    if sp.issparse(A): A = A.todense()
    if sp.issparse(B): B = B.todense()
    equal = np_almost_equal(A,B,epsilon = 1e-10)
    if not equal:
        print
        print A
        print 'is different from'
        print B
    return equal
