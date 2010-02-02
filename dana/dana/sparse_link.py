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
from functions import extract
from array import csr_array, dot
from link import link


class sparse_link(link):
    ''' '''

    def __init__(self, src, dst, kernel, name=None, dtype=np.double):
        '''
        Constructs a sparse link between source and destination using given kernel.
        
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
        src= self._src

        # Is kernel already a sparse array ?
        # -----------------------------------
        if sp.issparse(kernel):
            if kernel.shape != (dst.size,src.size):
                raise ValueError, \
                    'kernel shape is wrong relative to source and destination'
            else:
                self._kernel = csr_array(kernel, dtype=dtype)

        elif kernel.shape != (dst.size,src.size):
            if len(kernel.shape) == len(src.shape):
                K = sp.lil_matrix((dst.size,src.size), dtype=dtype)
                scale = src.size/float(dst.size)
                I = []
                J = []
                V = []
                Ks = np.array(list(kernel.shape), dtype=int)//2
                Ss = np.array(list(src.shape), dtype=int)//2
                for i in range(K.shape[0]):
                    index = np.array(list(np.unravel_index(i, dst.shape)))
                    index = (index/np.array(dst.shape, dtype=float)*np.array(src.shape)).astype(int)
                    k = extract(kernel, src.shape, Ks+Ss - index, 0).flatten()
                    J_ = k.nonzero()[0].tolist()
                    I_ = [i,]*len(J_)
                    V_ = k[J_].tolist()
                    I += I_
                    J += J_
                    V += V_
                K = sp.coo_matrix((V,(I,J)), shape=K.shape)
                self._kernel = csr_array(K, dtype=dtype)
            else:
                raise ValueError, \
                    'kernel shape is wrong relative to source and destination'
        else:
            self._kernel = csr_array(kernel, dtype=dtype)

        self.compute = self.compute_sum
        if name[-1] == '-':
            self.compute = self.compute_dst


    def compute(self):
        raise NotImplemented

    def compute_sum(self):
        src = self._src_data*self._src.mask
        dst = self._dst
        R = dot(self._kernel, src.reshape((src.size,1))).reshape(dst.shape)
        return R

    def compute_dst(self):
        src = self._src_data*self._src.mask
        dst = self._dst
        result = np.abs(self._kernel - src.reshape((1,src.size)))
        R = np.multiply((result.sum(axis=1).reshape(dst.shape)), dst.mask)
        return np.array(R,copy=False).reshape(dst.shape)

    def __getitem__(self, key):
        ''' '''
        key = np.array(key) % self._dst.shape
        s = np.ones((len(self._dst.shape),))
        for i in range(0,len(self._dst.shape)-1):
            s[i] = self._dst.shape[i]*s[i+1]
        index = int((s*key).sum()) 
        nz = self._kernel.mask[1][np.where(self._kernel.mask[0] == index)]
        a = np.ones_like(nz)*index
        nz = (a,nz)
        k = np.ones((self._src.size,))*np.NaN
        k[nz[1]] = self._kernel[nz]
        return k.reshape(self._src.shape)

if __name__ == '__main__':
    import numpy as np
    from group import group
    n = 3
    I = group((n,n))
    O = group((2*n+1,2*n+1))
    O.connect(I, np.ones((1,1)), 'L', sparse=True)
    print O.L[0,0]
    print O.L[-1,-1]
