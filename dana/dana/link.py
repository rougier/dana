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
''' A link represents a named connection between two groups.

Since groups may have several values, a link is actually a named connection
between a group and a specific value of another group:

>>> S = dana.zeros((3,3), keys=['V'])
>>> T = dana.zeros((3,3), keys=['V'])
>>> T.connect(S['V'], 'I', numpy.ones((1,1)))
>>> print T.get_weight(S,0,0)
array([[  1.,  NaN,  NaN],
       [ NaN,  NaN,  NaN],
       [ NaN,  NaN,  NaN]])
'''
import numpy as np
import scipy.sparse as sp
from functions import convolve1d, convolve2d, extract
from scipy.ndimage.interpolation import zoom
import scipy.linalg
from array import csr_array, dot
import inspect

epsilon = 1.0e-30


class link(object):
    ''' '''

    def __init__(self, source, destination, kernel, name=None,
                 dtype=np.float64, sparse=None, shared=False):
        ''' Construct a link between source and destination using given kernel.
        
        Parameters
        ----------
        source : group
            Source group.

        destination : group
            Destination group
           
        kernel : array or sparse array
            Kernel
            
        name : string
            group name

        dtype : data-type
            The desired data-type.
            
        sparse: True | False | None
           Indicate wheter internal storage should be sparse

        shared: True or False
            Whether the kernel is shared among elements composing the group.
            (only available for one-dimensional and two-dimensional groups)
        '''

        self.name = name
        if hasattr(source.dtype,'names') and source.dtype.names is not None:
            source = source[source.dtype.names[0]]
        self.source = source
        self.destination = destination
        self.shared = shared

        # Shared connection only works for 1d and 2d kernels and in these cases,
        # source and destination group must be of same dimension.
        if shared:
            # 1d convolution case
            # -------------------
            if len(source.shape) == len(destination.shape) == 1:
                if len(kernel.shape) == 1:
                    kernel = np.nan_to_num(kernel)
                    self.kernel = np.array(kernel, dtype=dtype)
                else:
                    raise ValueError, \
                        '''Shared link was requested but kernel shape does not match source's. '''

            # 2d convolution case
            # -------------------
            elif len(source.shape) == len(destination.shape) == 2:
                if len(kernel.shape) == 2:
                    # Pre-computing of singular value decomposition
                    kernel = np.nan_to_num(kernel)
                    self.USV = scipy.linalg.svd(kernel)
                    U,S,V = self.USV
                    self.USV = U.astype(dtype), S.astype(dtype), V.astype(dtype)
                    self.kernel = np.array(kernel, dtype=dtype)
                else:
                    raise ValueError, \
                        '''Shared link was requested but kernel shape does not match source's. '''
            # Unknown case
            # ------------
            else:
                raise ValueError, \
                    '''Shared link was requested but source dimension is > 2'''
            
        else:
            self.shared = False

            # Is kernel already a sparse array ?
            # -----------------------------------
            if sp.issparse(kernel):
                if kernel.shape != (destination.size,source.size):
                    raise ValueError, \
                        'kernel shape is wrong relative to source and destination'
                else:
                    self.kernel = csr_array(kernel, dtype=dtype)
                    #self.kernel = sp.csr_matrix(kernel)
                    #self.mask = self.kernel.nonzero()

            # Else, we expand kernel to the right size
            # ----------------------------------------
            else:
                # Replace all 0 with epsilon
                #kernel = np.where(kernel==0, epsilon, kernel)
                if kernel.shape != (destination.size,source.size):
                    if len(kernel.shape) == len(source.shape):
                        density = kernel.size/float(source.size*destination.size)
                        # The 0.25 threshold sparse/dense comes from benchmark-np.py
                        # You may have to change it according to benchmark results
                        if sparse: #or (density <= 0.25 and sparse is not False):
                            K = sp.lil_matrix((destination.size,source.size), dtype=dtype)
                            scale = source.size/float(destination.size)

                            I = []
                            J = []
                            V = []

                            Ks = np.array(list(kernel.shape), dtype=int)//2
                            Ss = np.array(list(source.shape), dtype=int)//2
                            for i in range(K.shape[0]):
                                index = np.array(list(np.unravel_index(i, destination.shape)))
                                index = (index/np.array(destination.shape, dtype=float)*np.array(source.shape)).astype(int)
                                k = extract(kernel, source.shape, Ks+Ss - index, 0).flatten()
                                #k = extract(kernel, source.shape, Ks+Ss - (index*scale).astype(int), 0).flatten()
                                J_ = k.nonzero()[0].tolist()
                                I_ = [i,]*len(J_)
                                V_ = k[J_].tolist()
                                I += I_
                                J += J_
                                V += V_
                            K = sp.coo_matrix((V,(I,J)), shape=K.shape)
                            self.kernel = csr_array(K, dtype=dtype)
                            #self.kernel = sp.csr_matrix(K)
                            #self.mask = self.kernel.nonzero()
                            return
                        else:
                            Ks = np.array(list(kernel.shape), dtype=int)//2
                            Ss = np.array(list(source.shape), dtype=int)//2
                            K = np.zeros((destination.size,source.size), dtype=dtype)
                            scale = source.size/float(destination.size)
                            for i in range(K.shape[0]):
                                index =  np.array(list(np.unravel_index(i, destination.shape)))
                                index = (index/np.array(destination.shape, dtype=float)*np.array(source.shape)).astype(int)
                                K[i,:] = extract(kernel, source.shape, Ks+Ss-index, np.NaN).flatten()
                                #K[i,:] = extract(kernel, source.shape, Ks+Ss-(index*scale).astype(int), np.NaN).flatten()
                            kernel = K
                    else:
                        raise ValueError, \
                            'kernel shape is wrong relative to source and destination'
                # The 0.25 threshold sparse/dense comes from benchmark-np.py
                # You may have to change it according to benchmark results
                # self.mask = np.isnan(kernel).astype(bool)

                K = np.nan_to_num(kernel)
                density = (K != 0).sum()/float(kernel.size)
                if sparse or (density <= 0.25 and sparse is not False):
                    self.kernel = csr_array(K, dtype=dtype)
                    #self.kernel = sp.csr_matrix(K, dtype=np.float32)
                    #self.mask = self.kernel.nonzero()
                else:
                    self.mask = np.isnan(kernel)
                    self.kernel = np.array(K, dtype=dtype)



    def __str__(self):
        ''' link.__str__() <==> str(x)'''

        if self.shared:
            shared='shared '
        else:
            shared= ''
        if sp.issparse(self.kernel):
            ktype = 'sparse'
        else:
            ktype = 'dense'
        src = self.source.parent.name
        dst = self.destination.name
        return '''<%s%sx%s %s link from group '%s' to group '%s'>''' % (
            shared, self.source.shape, self.destination.shape, ktype, src, dst)


    def __repr__(self):
        ''' link.__repr__() <==> repr(x)'''

        return self.__str__()
        

    def compute(self):
        ''' This method computes the link output depending on link type. If link
        name ends with '*', this method returns the weighted sum while if link
        name ends with '-', this methods returns the distance.
        '''

        if self.name[-1] == '*':
            return self.weighted_sum()
        elif self.name[-1] == '-':
            return self.distance()
        else:
            raise ValueError, 'link type is unknonw'


    def weighted_sum(self):
        ''' Return weighted sum of source and kernel. '''

        src  = np.nan_to_num(self.source)
        dst  = self.destination

        if len(src.shape) == len(dst.shape) == len(self.kernel.shape) == 1 and self.shared:
            s = np.array(dst.shape).astype(float)/src.shape
            S = convolve1d(zoom(src,s,order=1), self.kernel)
        elif len(src.shape) == len(dst.shape) == len(self.kernel.shape) == 2 and self.shared:
            s = np.array(dst.shape).astype(float)/src.shape
            S = convolve2d(zoom(src,s,order=1), self.kernel, self.USV)
        elif sp.issparse(self.kernel):
            S = dot(self.kernel,src.reshape((src.size,1))).reshape(dst.shape)
        else:
            S = np.dot(self.kernel,src.reshape((src.size,1))).reshape(dst.shape)
        return np.multiply(S,dst.mask)


    def distance(self):
        ''' Return distance between source and kernel. '''

        if self.shared:
            raise ValueError, 'Distance is not computable for shared links'
        src = self.source
        dst = self.destination

        if sp.issparse(self.kernel):
            result = np.abs(self.kernel - src.reshape((1,src.size)))
            return np.multiply((result.sum(axis=1).reshape(dst.shape)), dst.mask)
        else:
            result = np.abs(self.kernel - src.reshape((1,src.size)))*(1-self.mask)
            return np.multiply((result.sum(axis=1).reshape(dst.shape)), dst.mask)


    def learn(self, equation, links, dt=0.1, namespace=globals()):
        ''' Adapt link according to equation. '''

        src = self.source.parent
        dst = self.destination
        namespace['pre'] = src.reshape((1,src.size))
        namespace['post'] = dst.reshape((dst.size,1))
        namespace['W'] = self.kernel

        # Restore previously computed links
        for key in links.keys():
            locals()[key] = links[key].reshape((dst.size,1)) # + (1,1))

        if sp.issparse(self.kernel):
            R = eval(equation, namespace, locals())
            ij = self.kernel.mask
            R_ij = np.array(R[ij]).reshape(ij[0].size)
            K = sp.coo_matrix((R_ij,ij), self.kernel.shape)
            #self.kernel = sp.csr_matrix(K.tocsr())
            self.kernel = csr_array(K.tocsr())
        else:
            self.kernel[...] = eval(equation, namespace, locals())


    def __getitem__(self, key):
        ''' '''

        if not self.shared:
            s = np.ones((len(self.destination.shape),))
            for i in range(0,len(self.destination.shape)-1):
                s[i] = self.destination.shape[i]*s[i+1]
            index = int((s*key).sum())
            if sp.issparse(self.kernel):                
                nz = self.kernel.mask[1][np.where(self.kernel.mask[0] == index)]
                #nz = self.mask[1][np.where(self.mask[0] == index)]
                a = np.ones_like(nz)*index
                nz = (a,nz)
                k = np.ones((self.source.size,))*np.NaN
                k[nz[1]] = self.kernel[nz]
            else:
                k = self.kernel[index]*np.where(self.mask[index], np.NaN, 1)
            return k.reshape(self.source.shape)
        else:
            s = np.array(list(self.source.shape))/np.array(list(self.destination.shape))
            c = (np.array(list(key))*s).astype(int).flatten()
            Ks = np.array(list(self.kernel.shape), dtype=int)//2
            Ss = np.array(list(self.source.shape), dtype=int)//2
            return extract(self.kernel, self.source.shape, Ks+Ss-c, np.NaN)
