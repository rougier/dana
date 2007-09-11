//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "cube.h"

using namespace glpython::objects;


//_________________________________________________________________________Cube
Cube::Cube (std::string name) : core::Object (name)
{}

//________________________________________________________________________~Cube
Cube::~Cube (void)
{}

//_______________________________________________________________________render
void
Cube::render (void)
{
    float x, y, z, dx, dy, dz;
    x = y = z = 0.0f;
    dx = dy = dz = 1.0f;

    glPolygonOffset (1,1);
    glEnable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColor3f (1,0,0);
    cube (0,0,0,1,1,1);

    glDisable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glColor3f (0,0,0);    
    cube (0,0,0,1,1,1);
}

//_________________________________________________________________________cube
void
Cube::cube (float x, float y, float z, float dx, float dy, float dz)
{
    glPushMatrix ();
    glTranslatef (-(x+dx)/2, -(y+dy)/2, -(z+dz)/2);

    glBegin (GL_QUADS);
    // Top Face (-z)
    glNormal3f ( 0.0f,  0.0f,  1.0f);
    glVertex3f (    x,     y,  z+dz);
    glVertex3f ( x+dx,     y,  z+dz);
    glVertex3f ( x+dx,  y+dy,  z+dz);
    glVertex3f (    x,  y+dy,  z+dz);
    // Bottom Face (-z)
    glNormal3f ( 0.0f,  0.0f, -1.0f);
    glVertex3f (    x,     y,     z);
    glVertex3f (    x,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,     z);
    glVertex3f ( x+dx,     y,     z);
    // Front face (x)
    glNormal3f ( 1.0f,  0.0f,  0.0f);
    glVertex3f ( x+dx,     y,     z);
    glVertex3f ( x+dx,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,  z+dz);
    glVertex3f ( x+dx,     y,  z+dz);
    // Rear Face (-x)
    glNormal3f (-1.0f,  0.0f,  0.0f);
    glVertex3f (    x,     y,     z);
    glVertex3f (    x,     y,  z+dz);
    glVertex3f (    x,  y+dy,  z+dz);
    glVertex3f (    x,  y+dy,     z);
    // Left Face (y)
    glNormal3f ( 0.0f,  1.0f,  0.0f);
    glVertex3f (    x,     y,     z);
    glVertex3f (    x,     y,  z+dz);
    glVertex3f ( x+dx,     y,  z+dz);
    glVertex3f ( x+dx,     y,     z);
    // Right Face (-y)
    glNormal3f ( 0.0f, -1.0f,  0.0f);
    glVertex3f (    x,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,  z+dz);
    glVertex3f (    x,  y+dy,  z+dz);
    glEnd();
    glPopMatrix ();
}

//________________________________________________________________python_export
void
Cube::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Cube> >();

    class_<Cube, bases <Object> >("Cube",
        init< optional<std::string> >("__init__ ()"))
    ;       
}
