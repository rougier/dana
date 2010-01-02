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
import dana
from dana.array import csr_array, dot
from dana.tests.tools import np_equal, np_almost_equal
import numpy as np
import scipy.sparse as sp
from nose.tools import *

def test_add():
    ''' Check array addition '''

    A = np.random.random((5,10))+1
    B = np.random.random((5,10))+1
    C = np.random.random((5,1))
    D = np.random.random((10,))
    As = csr_array(A)
    Bs = csr_array(B)
    assert (np_almost_equal(A+1,As+1))
    assert (np_almost_equal(1+A,1+As))
    assert (np_almost_equal(A+B,As+B))
    assert (np_almost_equal(B+A,B+As))
    assert (np_almost_equal(A+C,As+C))
    assert (np_almost_equal(C+A,C+As))
    assert (np_almost_equal(A+D,As+D))
    assert (np_almost_equal(D+A,D+As))

def test_sub():
    ''' Check array subtract '''        

    A = np.random.random((5,10))+1
    B = np.random.random((5,10))+1
    C = np.random.random((5,1))
    D = np.random.random((10,))
    As = csr_array(A)
    Bs = csr_array(B)
    assert (np_almost_equal(A-1,As-1))
    assert (np_almost_equal(1-A,1-As))
    assert (np_almost_equal(A-B,As-B))
    assert (np_almost_equal(B-A,B-As))
    assert (np_almost_equal(A-C,As-C))
    assert (np_almost_equal(C-A,C-As))
    assert (np_almost_equal(A-D,As-D))
    assert (np_almost_equal(D-A,D-As))

def test_mul():
    ''' Check array multiplication '''        

    A = np.random.random((5,10))+1
    B = np.random.random((5,10))+1
    C = np.random.random((5,1))
    D = np.random.random((10,))
    As = csr_array(A)
    Bs = csr_array(B)
    assert (np_almost_equal(A*1,As*1))
    assert (np_almost_equal(1*A,1*As))
    assert (np_almost_equal(A*B,As*B))
    assert (np_almost_equal(B*A,B*As))
    assert (np_almost_equal(A*C,As*C))
    assert (np_almost_equal(C*A,C*As))
    assert (np_almost_equal(A*D,As*D))
    assert (np_almost_equal(D*A,D*As))

def test_div ():
    ''' Check array division '''        

    A = np.random.random((5,10))+1
    B = np.random.random((5,10))+1
    C = np.random.random((5,1))
    D = np.random.random((10,))
    As = csr_array(A)
    Bs = csr_array(B)
    assert (np_almost_equal(A/1,As/1))
    assert (np_almost_equal(1/A,1/As))
    assert (np_almost_equal(A/B,As/B))
    assert (np_almost_equal(B/A,B/As))
    assert (np_almost_equal(A/C,As/C))
    assert (np_almost_equal(C/A,C/As))
    assert (np_almost_equal(A/D,As/D))
    assert (np_almost_equal(D/A,D/As))

def test_dot():
    ''' Check array dot product '''        

    A = np.random.random((5,10))+1
    B = np.random.random((10,1))
    As = csr_array(A)
    Bs = csr_array(B)
    assert (np_almost_equal(np.dot(A,B), dot(As,B)))

def test_sum():
    ''' Check array sum '''        

    A = np.random.random((5,10))+1
    As = csr_array(A)
    assert (np_almost_equal(A.sum(), As.sum()))
    assert (np_almost_equal(A.sum(axis=0), As.sum(axis=0)))
    #assert (np_almost_equal(A.sum(axis=1), As.sum(axis=1)))


def test_misc():
    ''' Check array miscellaneous operations '''        

    A = np.random.random((5,10))+1
    As = csr_array(A)
    assert (np_almost_equal(np.cos(A), np.cos(As)))
    assert (np_almost_equal(np.sin(A), np.sin(As)))
    assert (np_almost_equal(np.exp(A), np.exp(As)))
    assert (np_almost_equal(np.sqrt(A),np.sqrt(As)))

def test_mask():
    ''' Check array mask '''

    A = np.zeros((5,5))
    A[0,0] = 1
    As = csr_array(A)
    A[1,0] = 1
    As += A
    assert (As[0,0] == 2)
    assert (As[1,0] == 0)
