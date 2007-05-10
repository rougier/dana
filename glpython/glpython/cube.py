#!/usr/bin/env python

#------------------------------------------------------------------------------
#
#   Copyright (c) 2007 Nicolas Rougier
# 
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
# 
#------------------------------------------------------------------------------
""" cube object """

from OpenGL.GL import *
from object import Object
from glpython.data import datadir

class Cube(Object):
    """ cube object """

    def __init__(self, x=0,y=0,z=0,dx=1,dy=1,dz=1):
        """ """
        
        Object.__init__(self)
        self.active = True
        self.visible = True
        self.size = (x,y,z,dx,dy,dz)
        self.dlist = 0


    def init (self):
        """ """

        if self.dlist:
            glDeleteLists (self.dlist, 1)
        self.dlist = glGenLists(1)

        x,y,z,dx,dy,dz = self.size

        glNewList (self.dlist, GL_COMPILE)
        n = [[0,0,1], [0,-1,0], [0,0,1], [0,1,0], [1,0,0], [-1,0,0]]
        faces = [[3,2,1,0], [7,6,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [3,0,4,7]]
        t = [[0,1], [1,1], [1,0], [0,0]]
        v = []
        for i in range(8):
            v.append([0,0,0])

        if dz < 0:
            z = dz
            dz = -dz

        v[0][0] = v[3][0] = v[4][0] = v[7][0] = x-dx / 2.0
        v[1][0] = v[2][0] = v[5][0] = v[6][0] = x+dx / 2.0
        v[2][1] = v[3][1] = v[6][1] = v[7][1] = y-dy / 2.0
        v[0][1] = v[1][1] = v[4][1] = v[5][1] = y+dy / 2.0
        v[4][2] = v[5][2] = v[6][2] = v[7][2] = z+dz / 2.0 #z
        v[0][2] = v[1][2] = v[2][2] = v[3][2] = z-dz / 2.0 #z+dz

        for i in range(6):
            glBegin (GL_QUADS)
            glNormal3fv (n[i])
            glVertex3fv (v[faces[i][0]])
            glVertex3fv (v[faces[i][1]])
            glVertex3fv (v[faces[i][2]])
            glVertex3fv (v[faces[i][3]])
            glEnd()
        glEndList()


    def render (self):
        """ """

        if not self.dlist:
            self.init()

        glPushAttrib (GL_ENABLE_BIT)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)
        glPolygonOffset (1,1)
        glEnable (GL_POLYGON_OFFSET_FILL)
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL)
        glColor3f (1,0,0)
        glMaterialfv (GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [1,0,0])
        glScalef (1,1,1)
        glCallList (self.dlist)

        glDisable (GL_POLYGON_OFFSET_FILL)
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE)
        glMaterialfv (GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0,0,0])
        glColor3f(0,0,0)
        glCallList (self.dlist)
        glPopAttrib ()

