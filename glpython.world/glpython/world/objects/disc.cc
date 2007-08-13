//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "disc.h"
#include <boost/python/detail/api_placeholder.hpp>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
using namespace glpython::world::objects;
using namespace boost::python;

//_________________________________________________________________________Disc
Disc::Disc (std::string name) : core::Object (name)
{
    x = 0;
    y = 0;
    z = 0;
    radius = 1.0f;
    phi = 90;
    theta = 0;
    color.append(0);
    color.append(1);
    color.append(0);    
}

//________________________________________________________________________~Disc
Disc::~Disc (void)
{}

//_______________________________________________________________________render
void
Disc::render (void)
{

    float r,g,b;
    r = extract<float>(color[0]);
    g = extract<float>(color[1]);
    b = extract<float>(color[2]);

    glPolygonOffset (1,1);
    glEnable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColor3f (r,g,b);
    disc ();

}

//_________________________________________________________________________disc
void
Disc::disc ()
{ 
    GLUquadric* params;
    params = gluNewQuadric();
    gluQuadricDrawStyle(params,GLU_FILL);

    glPushMatrix ();
    glTranslatef(x,y,z);

    glRotatef (-phi, 0, 1, 0);
    glRotatef (theta, 1, 0, 0);

    gluDisk(params,0,radius,32,1);
    glPopMatrix ();

    gluDeleteQuadric(params);

}

//________________________________________________________________python_export
void
Disc::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Disc> >();

    class_<Disc, bases <Object> >("Disc",
        init< optional<std::string> >("__init__ ()"))
        .def_readwrite ("x",     &Disc::x)
        .def_readwrite ("y",     &Disc::y)
        .def_readwrite ("z",     &Disc::z)
        .def_readwrite ("theta", &Disc::theta)
        .def_readwrite ("phi",     &Disc::phi)
        .def_readwrite ("radius", &Disc::radius)
        .def_readwrite ("color",     &Disc::color)
    ;       
}
