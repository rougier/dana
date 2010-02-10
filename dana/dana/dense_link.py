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
from functions import extract
from link import link


class dense_link(link):
    ''' '''

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

        link.__init__(self,src,dst,kernel,name,dtype)
        src = self._src
        dst = self._dst
        if kernel.shape != (dst.size,src.size):
            if len(kernel.shape) == len(src.shape):
                Ks = np.array(list(kernel.shape), dtype=int)//2
                Ss = np.array(list(src.shape), dtype=int)//2
                K = np.zeros((dst.size,src.size), dtype=dtype)
                scale = src.size/float(dst.size)
                for i in range(K.shape[0]):
                    index =  np.array(list(np.unravel_index(i, dst.shape)))
                    index = (index/np.array(dst.shape, dtype=float)*np.array(src.shape)).astype(int)
                    K[i,:] = extract(kernel, src.shape, Ks+Ss-index, np.NaN).flatten()
                self._mask = 1-np.isnan(K).astype(bool)
                self._kernel = np.nan_to_num(K)
            else:
                raise ValueError, \
                    'kernel shape is wrong relative to source and destination'
        else:
            self._kernel = kernel.astype(dtype)
            self._mask = 1-np.isnan(self._kernel).astype(bool)

        self._name = name
        self.compute = self.compute_sum
        if name[-1] == '-':
            self.compute = self.compute_dst


    def compile(self):
        name = self._name
        if self._src.mask.all():
            self.compute = self.compute_sum_no_mask
            if name[-1] == '-':
                self.compute = self.compute_dst_no_mask
        else:
            self.compute = self.compute_sum
            if name[-1] == '-':
                self.compute = self.compute_dst_no

    def compute_sum(self):
        src = self._src_data * self._src.mask
        dst = self._dst
        src = src.reshape((src.size,))
        R = np.dot(self._kernel,src)
        return R.reshape(dst.shape)

    def compute_sum_no_mask(self):
        src = self._src_data
        dst = self._dst
        src = src.reshape((src.size,))
        R = np.dot(self._kernel,src)
        return R.reshape(dst.shape)

    def compute_dst(self):
        src = self._src_data * self._src.mask
        dst  = self._dst
        result = np.abs(self._kernel - src.reshape((1,src.size)))*self._mask
        return np.multiply((result.sum(axis=1).reshape(dst.shape)), dst.mask)

    def compute_dst_no_mask(self):
        src  = self._src_data
        dst  = self._dst
        result = np.abs(self._kernel - src.reshape((1,src.size)))*self._mask
        return np.multiply((result.sum(axis=1).reshape(dst.shape)), dst.mask)

    def __getitem__(self, key):
        key = np.array(key) % self._dst.shape
        s = np.ones((len(self._dst.shape),))
        for i in range(0,len(self._dst.shape)-1):
            s[i] = self._dst.shape[i]*s[i+1]
        index = int((s*key).sum())
        k = self._kernel[index]*np.where(self._mask[index], 1, np.NaN)
        #k = self._kernel[index]
        return k.reshape(self._src.shape)


if __name__ == '__main__':
    import numpy as np
    from group import group
    n = 3
    I = group((n,n))
    O = group((2*n+1,2*n+1))
    O.connect(I, np.ones((1,1)), 'L')
    print O.L[0,0]
    print O.L[-1,-1]
