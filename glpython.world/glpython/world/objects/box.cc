//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "box.h"
#include <boost/python/detail/api_placeholder.hpp>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
using namespace glpython::world::objects;
using namespace boost::python;

//_________________________________________________________________________Box
Box::Box (std::string name) : core::Object (name)
{
    x = 0;
    y = 0;
    z = 0;
    length = 1.0;
    width = 1.0;
    height = 1.0;
    phi = 0;
    theta = 0;
    color.append(0);
    color.append(1);
    color.append(0);    
}

//________________________________________________________________________~Box
Box::~Box (void)
{}

//_______________________________________________________________________render
void
Box::render (void)
{

    float r,g,b;
    r = extract<float>(color[0]);
    g = extract<float>(color[1]);
    b = extract<float>(color[2]);

    glPolygonOffset (1,1);
    glEnable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColor3f (r,g,b);
    box ();
}

//_________________________________________________________________________box
void
Box::box ()
{ 
    glPushMatrix ();

    glTranslatef(x,y,z);
    glRotatef (theta, 1, 0, 0);
    glRotatef (-phi, 0, 1, 0);

    // SAMPLE VERTEX
    //glVertex3f ( width/2.0f, length/2.0f, height/2.0f);
    //glVertex3f ( width/2.0f, length/2.0f, height/2.0f);
    //glVertex3f ( width/2.0f, length/2.0f, height/2.0f);
    //glVertex3f ( width/2.0f, length/2.0f, height/2.0f);

    glBegin (GL_QUADS);
    // Top Face (-z)
    glNormal3f ( 0.0f, 0.0f,  1.0f);
    glVertex3f ( -width/2.0f, -length/2.0f,  height/2.0f);
    glVertex3f ( -width/2.0f, length/2.0f,  height/2.0f);
    glVertex3f ( width/2.0f, length/2.0f,  height/2.0f);
    glVertex3f ( width/2.0f, -length/2.0f,  height/2.0f);
    // Bottom Face (-z)
    glNormal3f ( 0.0f,  0.0f, -1.0f);
    glVertex3f ( -width/2.0f, -length/2.0f, -height/2.0f);
    glVertex3f ( -width/2.0f, length/2.0f, -height/2.0f);
    glVertex3f ( width/2.0f, length/2.0f, -height/2.0f);
    glVertex3f ( width/2.0f, -length/2.0f, -height/2.0f);
    // Front face (x)
    glNormal3f ( 1.0f,  0.0f,  0.0f);
    glVertex3f ( width/2.0f, -length/2.0f, -height/2.0f);
    glVertex3f ( width/2.0f, -length/2.0f, height/2.0f);
    glVertex3f ( width/2.0f, length/2.0f, height/2.0f);
    glVertex3f ( width/2.0f, length/2.0f, -height/2.0f);
    // Rear Face (-x)
    glNormal3f (-1.0f,  0.0f,  0.0f);
    glVertex3f ( -width/2.0f, -length/2.0f, -height/2.0f);
    glVertex3f ( -width/2.0f, -length/2.0f, height/2.0f);
    glVertex3f ( -width/2.0f, length/2.0f, height/2.0f);
    glVertex3f ( -width/2.0f, length/2.0f, -height/2.0f);
    // Left Face (y)
    glNormal3f ( 0.0f,  1.0f,  0.0f);
    glVertex3f ( -width/2.0f, length/2.0f, -height/2.0f);
    glVertex3f ( -width/2.0f, length/2.0f, height/2.0f);
    glVertex3f ( width/2.0f, length/2.0f, height/2.0f);
    glVertex3f ( width/2.0f, length/2.0f, -height/2.0f);
    // Right Face (-y)
    glNormal3f ( 0.0f, -1.0f,  0.0f);
    glVertex3f ( -width/2.0f, -length/2.0f, -height/2.0f);
    glVertex3f ( -width/2.0f, -length/2.0f, height/2.0f);
    glVertex3f ( width/2.0f, -length/2.0f, height/2.0f);
    glVertex3f ( width/2.0f, -length/2.0f, -height/2.0f);
    glEnd();
    glPopMatrix ();

}

//________________________________________________________________python_export
void
Box::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Box> >();

    class_<Box, bases <Object> >("Box",
        init< optional<std::string> >("__init__ ()"))
        .def_readwrite ("x",     &Box::x)
        .def_readwrite ("y",     &Box::y)
        .def_readwrite ("z",     &Box::z)
        .def_readwrite ("length", &Box::length)
        .def_readwrite ("height",     &Box::height)
        .def_readwrite ("width", &Box::width)
        .def_readwrite ("theta", &Box::theta)
        .def_readwrite ("phi", &Box::phi)
        .def_readwrite ("color",     &Box::color)
    ;       
}
