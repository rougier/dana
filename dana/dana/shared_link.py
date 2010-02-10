#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# DANA, Distributed Asynchronous Adaptive Numerical Computing Framework
# Copyright (c) 2009, 2010 Nicolas Rougier - INRIA - CORTEX Project
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
from functions import convolve1d, convolve2d, extract
from scipy.ndimage.interpolation import zoom
import scipy.linalg
from link import link


class shared_link(link):

    def __init__(self, src, dst, kernel, name=None, dtype=np.double):
        '''
        Constructs a shared link between source and destination using given kernel.
        
        **Parameters**

        src : (group, field)
            Source group and field
        dst : group
            Destination group
        kernel : array-like
            Kernel
        name : string
            link name
        dtype : data-type
            Desired data-type.
        '''

        link.__init__(self, src, dst, kernel, name, dtype)
        self._kernel = np.array(kernel, dtype=dtype)
        self._scale = np.array(self._dst.shape).astype(float)/self._src.shape

        # 1d convolution case
        # -------------------
        if len(self._src.shape) == len(self._dst.shape) == 1:
            if len(kernel.shape) != 1:
                raise ValueError, \
                 '''Shared link requested but kernel shape does not match.'''
            self.compute = self._compute_1

        # 2d convolution case
        # -------------------
        elif len(self._src.shape) == len(self._dst.shape) == 2:
            if len(kernel.shape) != 2:
                raise ValueError, \
                    '''Shared link requested but kernel shape does not match.'''
            self._USV = scipy.linalg.svd(self._kernel)
            U,S,V = self._USV
            self._USV = U.astype(dtype), S.astype(dtype), V.astype(dtype)
            self._scale = np.array(self._dst.shape).astype(float)/self._src.shape
            self.compute = self._compute_2

        # Unknown case
        # ------------
        else:
            raise ValueError, \
                '''Shared link requested but dimensions are too high (> 2).'''
 
    def compile(self):
        if self._src.mask.all():
            if len(self._kernel.shape) == 1:
                self.compute = self._compute_1_no_mask
            elif len(self._kernel.shape) == 2:
                self.compute = self._compute_2_no_mask
        else:
            if len(self._kernel.shape) == 1:
                self.compute = self._compute_1
            elif len(self._kernel.shape) == 2:
                self.compute = self._compute_2

    def _compute_1(self):
        ''' One dimensional convolution '''

        src = self._src_data * self._src.mask
        dst = self._dst
        return convolve1d(zoom(src, self._scale,order=1), self._kernel)

    def _compute_2(self):
        ''' Two dimensional convolution '''

        src = self._src_data * self._src.mask
        dst = self._dst
        return convolve2d(zoom(src, self._scale,order=1), self._kernel, self._USV)


    def _compute_1_no_mask(self):
        ''' One dimensional convolution '''

        src = self._src_data
        dst = self._dst
        return convolve1d(zoom(src, self._scale,order=1), self._kernel)

    def _compute_2_no_mask(self):
        ''' Two dimensional convolution '''

        src = self._src_data
        dst = self._dst
        return convolve2d(zoom(src, self._scale,order=1), self._kernel, self._USV)


    def __getitem__(self, key):
        key = np.array(key) % self._dst.shape
        s = np.array(list(self._src.shape)).astype(float)/np.array(list(self._dst.shape))
        c = (key*s).astype(int).flatten()
        Ks = np.array(list(self._kernel.shape), dtype=int)//2
        Ss = np.array(list(self._src.shape), dtype=int)//2
        return extract(self._kernel, self._src.shape, Ks+Ss-c, np.NaN)


if __name__ == '__main__':
    import numpy as np
    from group import group

    n = 3
    I = group((n,n))
    O = group((2*n+1,2*n+1))
    O.connect(I, np.ones((1,1)), 'L', shared=True)
    print O.L[0,0]
    print O.L[-1,-1]


    G1 = group((10,))
    G1.V = 2.5
    G2 = group((10,))
    G2.V = 1.1
    G1.connect(G2, np.ones((1,)), 'I', sparse=True)
    G1.dI = 'post.V*pre.V-W'
    G1.compile()
    G1.compute()
    G1.learn()
    print G1.I._kernel
    print np.eye(10)*2.75
