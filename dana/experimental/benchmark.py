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
import time, numpy, dana


def benchmark(size=50, iterations=100, kernel = numpy.ones((1,1)), epsilon = 1e-10):
    n = size
    src = dana.zeros((n,n))
    dst = dana.zeros((n,n))
    src.V = numpy.random.random((n,n))

    dst.connect(src.V, kernel, 'I', shared=False, sparse=False)
    dst.V = 0
    dst.dV = 'I'
    t = time.clock()
    for i in range(iterations):
        dst.compute()
    t1 = time.clock()-t
    dst1 = dst.V.copy()

    dst.connect(src.V, kernel, 'I', shared=True, sparse=False)
    dst.V = 0
    dst.dV = 'I'
    t = time.clock()
    for i in range(iterations):
        dst.compute()
    t2 = time.clock()-t
    dst2 = dst.V.copy()

    dst.connect(src.V, kernel, 'I', shared=False, sparse=True)
    dst.V = 0
    dst.dV = 'I'
    t = time.clock()
    for i in range(iterations):
        dst.compute()
    t3 = time.clock()-t
    dst3 = dst.V.copy()

    if (not numpy.all(numpy.abs(dst1-dst2) < epsilon) or
        not numpy.all(numpy.abs(dst1-dst3) < epsilon) or
        not numpy.all(numpy.abs(dst2-dst3) < epsilon)):
        print dst1-dst2
        raise RuntimeError, 'Results are not equal'
    return (t1,t2,t3)


n = 50
iterations = 100
print 'Source size:      %dx%d' % (n,n)
print 'Destination size: %dx%d' % (n,n)
print
for i in [1,3,25,50,100]:
    t1,t2,t3 = benchmark(size=n, iterations=iterations,kernel=numpy.ones((i,i)))
    print '%dx%d kernel, constant' % (i,i)
    print 'dense  : %.1f ms' % (t1/iterations*1000)
    print 'shared : %.1f ms' % (t2/iterations*1000)
    print 'sparse : %.1f ms' % (t3/iterations*1000)
    print
    t1,t2,t3 = benchmark(size=n, kernel=numpy.random.random((i,i)))
    print '%dx%d kernel, random' % (i,i)
    print 'dense  : %.1f ms' % (t1/iterations*1000)
    print 'shared : %.1f ms' % (t2/iterations*1000)
    print 'sparse : %.1f ms' % (t3/iterations*1000)
    print

