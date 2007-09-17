#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2007 Nicolas Rougier.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$
#------------------------------------------------------------------------------
""" Various GLPython objects

"""

import OpenGL.GL as GL
import glpython.core
from _objects import *
import os.path


class Logo (glpython.core.Object):
    """ Python 3D logo """

    def __init__(self):
        """ """

        glpython.core.Object.__init__(self)
        self.dlist = 0

    
    def load (self, filename):
        """ Load geometry from a wavefront .obj file """

        list = GL.glGenLists (1)
        GL.glNewList (list, GL.GL_COMPILE)
        file = open (filename)
        vertices = []
        normals  = []
        faces    = []
        for line in file:
            fields = line.split(' ')
            if fields[0] == 'v':
                x,y,z = float(fields[1]), float(fields[2]), float(fields[3])
                vertices.append ([x,y,z])
            elif fields[0] == 'vn':
                x,y,z = float(fields[1]), float(fields[2]), float(fields[3])
                normals.append ([x,y,z])
            elif fields[0] == 'f':
                if len(fields) == 3:
                    GL.glBegin (GL.GL_LINES)
                elif len(fields) == 4:
                    GL.glBegin (GL.GL_TRIANGLES)
                elif len(fields) == 5:
                    GL.glBegin (GL.GL_QUADS)
                for f in fields[1:]:
                    if f.count('//'):
                        f = f.split('//')
                        v,n = int(f[0]), int(f[1]) 
                        GL.glVertex3fv (vertices[v-1])
                        GL.glNormal3fv (normals[n-1])
                    else:
                        v = int(f)
                        GL.glVertex3fv (vertices[v-1])
                GL.glEnd ()
        file.close()
        GL.glEndList()
        return list


    def build (self):
        """ Build logo from file """

        dir = os.path.dirname (__file__)
        l1 = self.load (os.path.join (dir, "python-logo.obj"))
        l2 = self.load (os.path.join (dir, "python-logo-inner-line.obj"))
        l3 = self.load (os.path.join (dir, "python-logo-outer-line.obj"))
        l4 = self.load (os.path.join (dir, "python-logo-border-line.obj"))

        self.dlist = GL.glGenLists (1)
        GL.glNewList (self.dlist, GL.GL_COMPILE)
        
        GL.glLineWidth (1.0)
        GL.glPushMatrix()

        GL.glDisable(GL.GL_BLEND)
        GL.glTranslatef (-2,0,-2)
        GL.glPolygonOffset (1,1)
        GL.glEnable (GL.GL_POLYGON_OFFSET_FILL)
        GL.glColor ([255/255.0, 232/255.0, 115/255.0])
        GL.glCallList (l1)

        GL.glDisable (GL.GL_POLYGON_OFFSET_FILL)
        GL.glColor ( (1,1,1) )
        GL.glCallList (l2)
        GL.glCallList (l3)
        GL.glPushMatrix ()
        GL.glTranslatef (0,-4,0)
        GL.glCallList (l2)
        GL.glCallList (l3)
        GL.glPopMatrix ()
        GL.glCallList (l4)

        GL.glScalef (-1,1,-1)
        GL.glTranslatef (-4,0,-4)

        GL.glEnable (GL.GL_POLYGON_OFFSET_FILL)
        GL.glColor ([90/255.0, 159/255.0, 212/255.0])
        GL.glCallList (l1)
        GL.glDisable (GL.GL_POLYGON_OFFSET_FILL)
        GL.glColor ( (1,1,1) )
        GL.glCallList (l2)
        GL.glCallList (l3)
        GL.glCallList (l4)
        GL.glPushMatrix ()
        GL.glTranslatef (0,-4,0)
        GL.glCallList (l2)
        GL.glCallList (l3)
        GL.glPopMatrix ()
        GL.glPopMatrix()
        GL.glLineWidth (1.0)
        GL.glEndList ()

    
    def render (self):
        """ Render logo """

        GL.glPushMatrix()
        GL.glScalef (.1,.1,.1)
        if not self.dlist:
            self.build()
        GL.glCallList (self.dlist)
        GL.glPopMatrix()
