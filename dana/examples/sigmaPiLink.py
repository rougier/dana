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
''' A sigmaPilink represents a named connection between three groups.

'''
import numpy as np
from dana.functions import extract
from dana.link import link

class sigmaPiLink(link):
    ''' '''

    def __init__(self, first_source, second_source, destination, alpha, beta, delta, type, weight, name=None,
                 dtype=np.double):
        ''' Construct a sigmapi link between two sources and one destination
        The connections are defined such that the following relationship holds :
        alpha * first_source + beta * destination + delta = second_source
        In this highly specific implementation, alpha, beta, delta have to be integers
        
        **Parameters**

        first_source, second_source : the two source groups
        destination                 : destination group
        alpha, beta, delta          : parameters for these specific sigma-pi links
        type                        : variable used to know if we use optimized links, and which one
        weight (scalar)             : scaling factor of all the weights
        '''
	#
	# Incompatible parameters for the constructor of link ??
	# 
	#link.__init__(self, None, None, None, name, dtype, False, False)
        self.source = first_source
	self.first_source = first_source
	self.second_source = second_source
	self.destination = destination
	self.alpha = alpha
	self.beta = beta
	self.delta = delta
        self.type = type
	self.weight = weight
        self._dst = destination

    def __str__(self):
        ''' link.__str__() <==> str(x)'''
	
	# What is group.base ???
        first_src = self.first_source.base.name
	second_src = self.second_source.name
        dst = self.destination.name
        return '''<%sx%sx%s sigma-pi link from group '%s' and group '%s' to group '%s'>''' % (
            self.first_source.shape , self.second_source.shape,
            destination.shape, ktype, first_src,
            second_src              , dst)
    

    def __repr__(self):
        ''' link.__repr__() <==> repr(x)'''
        return self.__str__()
  
    def compile(self):
        '''Nothing to do'''

    def compute(self):
        ''' This method computes the output of a sigma-pi link '''
        
        res = np.zeros(self.destination.shape)
        if self.type == 0:
            # In case the source maps are 1D
            if(len(self.first_source.shape) == len(self.second_source.shape) == 1):
                # Version specific for alpha = 1, beta = -1 , delta = N/2
                for k in range(self.first_source.shape[0]):
                    # the shape/2 offset is because extract takes the position of the center of the matrix
                    extracted_second_source = extract(self.second_source, shape = self.second_source.shape, position = (self.second_source.shape[0]/2+self.beta*k+self.delta, ), fill = 0.0)
                    res += self.first_source[k] * extracted_second_source
                res = res * self.weight
            else:
                # Version specific for alpha = 1, beta = -1 , delta = N/2
                for k in range(self.first_source.shape[0]):
                    for l in range(self.first_source.shape[1]):
                        # the shape/2 offset is because extract takes the position of the center of the matrix
                        extracted_second_source = extract(self.second_source, shape = self.second_source.shape, position = (self.second_source.shape[0]/2+self.beta*k+self.delta, self.second_source.shape[1]/2+self.beta*l+self.delta), fill = 0.0)
                        res += self.first_source[k,l] * extracted_second_source
                res = res * self.weight                
        elif self.type == 1:
            # In case the source maps are 1D
            if(len(self.first_source.shape) == len(self.second_source.shape) == 1):
                # Version specific for alpha = -1, beta = 1, delta = N/2
                for k in range(self.first_source.shape[0]):
                    extracted_second_source = extract(self.second_source, shape = self.second_source.shape, position = (self.beta*k,), fill = 0.0)
                    res += self.first_source[k] * np.flipud(extracted_second_source)
                res = res * self.weight
            else:                
                # Version specific for alpha = -1, beta = 1, delta = N/2
                for k in range(self.first_source.shape[0]):
                    for l in range(self.first_source.shape[1]):
                        extracted_second_source = extract(self.second_source, shape = self.second_source.shape, position = (self.beta*k, self.beta*l), fill = 0.0)
                        res += self.first_source[k,l] * np.fliplr(np.flipud(extracted_second_source))
                res = res * self.weight             
        else:
            # In case the source maps are 1D
            if(len(self.first_source.shape) == len(self.second_source.shape) == 1):
                for i in range(self.destination.shape[0]):
                    if(self.beta > 0):
                        low_k = int(max(0, (-self.alpha * i - self.delta)/self.beta))
                        high_k = int(min(self.first_source.shape[0] - 1, (self.second_source.shape[0] - 1 - self.alpha * i - self.delta)/self.beta))
                    else:
                        low_k = int(max(0,(self.second_source.shape[0] - 1 - self.alpha * i - self.delta)/self.beta))
                        high_k = int(min(self.first_source.shape[0] - 1,(-self.alpha * i - self.delta)/self.beta))
                    for k in range(low_k, high_k + 1):
                        m = int(self.alpha * i + self.beta * k + self.delta)
                        res[i] += self.weight* self.first_source[k] * self.second_source[m]
            else:
                # This version is slightly optimized for the bounds involved in the convolution
                for i in range(self.destination.shape[0]):
                    for j in range(self.destination.shape[1]):
                        if(self.beta > 0):
                            low_k = int(max(0, (-self.alpha * i - self.delta)/self.beta))
                            high_k = int(min(self.first_source.shape[0] - 1, (self.second_source.shape[0] - 1 - self.alpha * i - self.delta)/self.beta))
                            low_l = int(max(0, (-self.alpha * j - self.delta)/self.beta))
                            high_l = int(min(self.first_source.shape[1] - 1, (self.second_source.shape[1] - 1 - self.alpha * j - self.delta)/self.beta))
                        else:
                            low_k = int(max(0,(self.second_source.shape[0] - 1 - self.alpha * i - self.delta)/self.beta))
                            high_k = int(min(self.first_source.shape[0] - 1,(-self.alpha * i - self.delta)/self.beta))
                            low_l = int(max(0,(self.second_source.shape[1] - 1 - self.alpha * j - self.delta)/self.beta))
                            high_l = int(min(self.first_source.shape[1] - 1,(-self.alpha * j - self.delta)/self.beta))
                        for k in range(low_k, high_k + 1):
                            for l in range(low_l, high_l + 1):
                                m = int(self.alpha * i + self.beta * k + self.delta)
                                n = int(self.alpha * j + self.beta * l + self.delta)
                                res[i,j] += self.weight* self.first_source[k,l] * self.second_source[m,n]
        # Non optimized version
        #         res = np.zeros(self.destination.shape)
        #         for i in range(self.destination.shape[0]):
        #             for j in range(self.destination.shape[1]):
        #                 for k in range(self.first_source.shape[0]):
        #                     for l in range(self.first_source.shape[1]):
        #                         m = int(self.alpha * i + self.beta * k + self.delta)
        #                         n = int(self.alpha * j + self.beta * l + self.delta)
        #                         if((m >= 0) and (m < self.second_source.shape[0])and
        #                            (n >= 0) and (n < self.second_source.shape[1])):
        #                            res[i,j] += self.weight* self.first_source.V[k,l] * self.second_source._values['V'][m,n]
        
	return res;

    def learn(self, equation, links, dt=0.1, namespace=globals()):
        ''' Adapt link according to equation. '''
        return

    def __getitem__(self, key):
        ''' '''
	return np.zeros(self.first_source.shape)
