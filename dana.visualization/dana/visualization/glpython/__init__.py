#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: __init__.py 145 2007-05-10 14:18:42Z rougier $
#------------------------------------------------------------------------------
""" GLPython visualization tools

"""

import glpython
from dana.visualization.glpython.network import Network
from dana.visualization.glpython.weights import Weights

class Figure (glpython.Figure):
    """
    Figure class
    """
   
    def __init__ (self, **kwargs):
        """
        Create a new empty figure
        
        Function signature
        ------------------
        
        __init__ (...) 

        Keyword arguments
        -----------------        
        
        size  -- Relative or absolute size
        
        position -- Relative or absolute position
        
        has_border -- Whether figure has border or not
        """

        glpython.Figure.__init__ (self, **kwargs)
        

    def network (self, net,
                 cmap = glpython.CM_Default, style = 'flat', title = None,
                 show_colorbar = True, show_label = True):
        """
        
        Create a view for a DANA network
        
        Function signature
        ------------------
        
        network (net, figure, ...) 

        Function signature
        ------------------

        net -- A DANA network

        Keyword arguments
        -----------------        
        
        cmap  -- Colormap to use
        
        style -- 'flat' or 'surface'
        
        title -- Firue title
        
        show_colorbar -- Whether to display colorbar
        
        show_labels -- Wheter to display map names
        
        """
        
        net = Network (net, self, cmap=cmap, style = style, title=title,
                       show_colorbar = show_colorbar,
                       show_label=show_label)
        self.append (net)
        return net

    def weights (self, src, dst,
                 cmap = glpython.CM_Default, title = None,
                 show_colorbar = True, show_label = True):
        """
        
        Create a view for weights going from src to dst
        
        Function signature
        ------------------
        
        network (net, src, dst, ...) 

        Function signature
        ------------------

        net -- A DANA network
        
        src -- Source layer
        
        dst -- Destination layer

        Keyword arguments
        -----------------        
        
        cmap  -- Colormap to use
        
        title -- Title
        
        show_colorbar -- Whether to display colorbar
        
        show_labels -- Wheter to display map names
        
        """
        
        w = Weights (src, dst, self, cmap=cmap, title=title,
                       show_colorbar = show_colorbar,
                       show_label=show_label)
        self.append (w)
        return w

