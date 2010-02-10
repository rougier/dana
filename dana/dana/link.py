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


class link(object):

    def __init__(self, src, dst, kernel, name=None, dtype=np.double):
        '''
        Constructs a link between source and destination using given kernel.
        
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

        self._name = name
        if type(src) in [tuple,list]:
            self._src = src[0]
            self._src_data = getattr(src[0],src[1])
        else:
            self._src = src
            key = src.dtype.names[0]
            self._src_data = getattr(src,key)
        self._dst = dst
        self._kernel = None
 
    def compute(self):
        raise NotImplemented

    def compile(self):
        pass
