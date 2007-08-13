//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "bar.h"
#include <boost/python/detail/api_placeholder.hpp>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
using namespace glpython::world::objects;
using namespace boost::python;

//_________________________________________________________________________Bar
Bar::Bar (std::string name) : core::Object (name)
{
    x = 0;
    y = 0;
    z = 0;
    length = 1.0;
    height = 0.2;
    phi = 0;
    theta = 0;
    color.append(0);
    color.append(1);
    color.append(0);    
}

//________________________________________________________________________~Bar
Bar::~Bar (void)
{}

//_______________________________________________________________________render
void
Bar::render (void)
{

    float r,g,b;
    r = extract<float>(color[0]);
    g = extract<float>(color[1]);
    b = extract<float>(color[2]);

    glPolygonOffset (1,1);
    glEnable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColor3f (r,g,b);
    bar ();

}

//_________________________________________________________________________bar
void
Bar::bar ()
{ 
    glPushMatrix ();

    glTranslatef(x,y,z);
    glRotatef (theta, 1, 0, 0);
    glRotatef (-phi, 0, 1, 0);

    glBegin (GL_QUADS);
    glNormal3f ( 1.0f, 0.0f, 0.0f);
    glVertex3f ( 0, -length/2.0f, -height/2.0f);
    glVertex3f ( 0, length/2.0f,  -height/2.0f);
    glVertex3f ( 0, length/2.0f, height/2.0f);
    glVertex3f ( 0, -length/2.0f, height/2.0f);
    glEnd();

    glPopMatrix ();

}

//________________________________________________________________python_export
void
Bar::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Bar> >();

    class_<Bar, bases <Object> >("Bar",
        init< optional<std::string> >("__init__ ()"))
        .def_readwrite ("x",     &Bar::x)
        .def_readwrite ("y",     &Bar::y)
        .def_readwrite ("z",     &Bar::z)
        .def_readwrite ("length", &Bar::length)
        .def_readwrite ("height",     &Bar::height)
        .def_readwrite ("theta", &Bar::theta)
        .def_readwrite ("phi", &Bar::phi)
        .def_readwrite ("color",     &Bar::color)
    ;       
}
