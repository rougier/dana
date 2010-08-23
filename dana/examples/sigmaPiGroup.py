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
from array import array
from numpy import minimum, maximum, sin, cos, exp, sqrt, multiply, dot

from dana.functions import extract, convolve1d, convolve2d

from dana.group import group
from dana.link import link

import sigmaPiLink
import inspect


class sigmaPiGroup(group):
    ''' A group represents a vector of homogeneous elements.

    A sigmaPiGroup is a dana.group to which a specific method is added for sigma-pi links.
    '''

    def __init__(self, shape=(), dtype=np.double, keys=['V'],
                 mask=True, fill=None, name=''):
        ''' Create a group.
        
        **Parameters**

            shape : tuple of integer or array-like object
                Shape of output array or object to create group from.
            dtype : data-type
                Desired data-type.
            keys : list of strings
                Names of the different values
            mask : boolean or boolean array
                boolean mask indicating active and inactive elements
            name : string
                Group name
            fill : scalar or scalar array
                Fill value

        **Returns**
            out: group
                Group of given shape and type.
        '''
        group.__init__(self,shape,dtype,keys,mask,fill,name)

    def connect_sigmapi(self, first_source ,second_source, alpha, beta, delta, weight, name, dtype=np.double):
        '''
        Create a sigma-pi connection between three groups.
        If the name ends by * or #, the connection involves predefined specific methods
        for computing the value of the link. Otherwise, the connectivity pattern used reads :
        destination.position = alpha * first_source.position + beta * second_source.position + delta

        **Parameters**

            first_source, second_source : groups
                 The two afferent groups involved in the connection
                 
            alpha, beta, delta : int
                 Parameters of the sigma-pi connections
                 
            weight :
                Weight of the connection
                
            name : string
                Name of the connection

            dtype : data-type
                Desired data-type.
        '''
        if name[-1] == '*':
            self._link[name[:-1]] = sigmaPiLink.sigmaPiLink(first_source = first_source, second_source = second_source, 
                                                             destination = self  , alpha = 1.0, 
                                                             beta = -1.0         , delta = second_source.shape[0]/2,
                                                             type = 0            , weight = weight,
                                                             name = name[:-1]    , dtype = dtype)
        elif name[-1] == '#':
            self._link[name[:-1]] = sigmaPiLink.sigmaPiLink(first_source = first_source, second_source = second_source, 
                                                             destination = self  , alpha = -1.0, 
                                                             beta = 1.0          , delta = second_source.shape[0]/2,
                                                             type = 1            , weight = weight,
                                                             name = name[:-1]    , dtype = dtype)
        else:
            self._link[name] = sigmaPiLink.sigmaPiLink(first_source = first_source, second_source = second_source, 
                                                             destination = self  , alpha = alpha, 
                                                             beta = beta         , delta = delta,
                                                             type = 2            , weight = weight,
                                                             name = name         , dtype = dtype)
