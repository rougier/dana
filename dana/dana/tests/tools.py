#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009 Nicolas Rougier - INRIA - CORTEX Project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either  version 3 of the  License, or (at your  option)
# any later version.
# 
# This program is  distributed in the hope that it will  be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR  A  PARTICULAR PURPOSE.  See  the GNU  General  Public 
# License for  more details.
# 
# You should have received a copy  of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
# 
# Contact:  CORTEX Project - INRIA
#           INRIA Lorraine, 
#           Campus Scientifique, BP 239
#           54506 VANDOEUVRE-LES-NANCY CEDEX 
#           FRANCE
import numpy as np
import scipy.sparse as sp

def np_almost_equal(A, B, epsilon=1e-10):
    ''' Assert two arrays are almost equal, even with NaN in them '''

    if sp.issparse(A):
        A = A.todense()
    if sp.issparse(B):
        B = B.todense()
    A_nan = np.isnan(A)
    B_nan = np.isnan(B)
    A_num = np.nan_to_num(A)
    B_num = np.nan_to_num(B)
    return np.all(A_nan==B_nan) and (abs(A_num-B_num)).sum() <= epsilon

def np_equal(A, B):
    ''' Assert two arrays are equal, even with NaN in them '''

    if sp.issparse(A):
        A = A.todense()
    if sp.issparse(B):
        B = B.todense()
    return np_almost_equal(A,B,epsilon = 0)
