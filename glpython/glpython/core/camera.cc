//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <GL/gl.h>
#include <GL/glu.h>
#include "camera.h"

using namespace glpython::core;


//_______________________________________________________________________Camera
Camera::Camera (std::string name) :
        name(name), near(1), far(100), aperture(30), zoom(1), is_ortho(false)
{
    sx = sy = 0;
    w = h = 1;
}

//______________________________________________________________________~Camera
Camera::~Camera (void)
{}

//_________________________________________________________________________push
void
Camera::push (void)
{
    int mode;
    glMatrixMode (GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity ();
    glGetIntegerv (GL_RENDER_MODE, &mode);

    if (mode == GL_SELECT) {
        GLint viewport[4];
        glGetIntegerv (GL_VIEWPORT, viewport);
        gluPickMatrix (this->sx, this->sy, 1, 1, viewport);
    }

    float aspect, left, right, bottom, top;
    aspect = 1.0f;
    left = right = bottom = top = 0.0f;
    if (this->h == 0) {
        aspect = float(this->w);
    } else {
        aspect = this->w/float(this->h);
    }

    if (is_ortho) {
        if (aspect > 1.0) {
            left = -aspect;
            right = aspect;
            bottom = -1.0f;
            top = 1.0f;
        } else {
            left = -1.0f;
            right =  1.0f;
            bottom = -1.0/aspect;
            top =  1.0/aspect;
        }
        glOrtho (left, right, bottom, top, near, far);
    } else {
        gluPerspective (aperture, aspect, near, far);
    }
    
    glMatrixMode (GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity ();
    glTranslatef (0.0f, 0.0f, -5.0f);
    if (zoom < 0.1)
        zoom = 0.1;
    else if (zoom > 5)
        zoom = 5;
    glScalef (zoom*2, zoom*2, zoom*2);
}

//__________________________________________________________________________pop
void
Camera::pop (void)
{
    glMatrixMode (GL_PROJECTION);
    glPopMatrix ();
    glMatrixMode (GL_MODELVIEW);
    glPopMatrix();
}

//_________________________________________________________________select_event
void
Camera::select_event (int x, int y)
{
    this->sx = x;
    this->sy = y;
}

//_________________________________________________________________________repr
std::string
Camera::repr (void)
{
    return name;
}

//________________________________________________________________python_export
void
Camera::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Camera> >();

    class_<Camera>("Camera",
    "======================================================================\n"
    "                                                                      \n"
    " A camera describes the view matrix used for rendering a scene.       \n"
    " Attributes:                                                          \n"
    "   name - name of the camera                                          \n"
    "   is_ortho - whether camera is in ortho mode                         \n"
    "   zoom - zoom factor                                                 \n"
    "   near,far - near and far planes                                     \n"
    "   aperture - objective aperture                                      \n"
    "                                                                      \n"
    "======================================================================\n",
     
    init< optional<std::string> >("__init__ ()"))
        
    .def_readwrite ("name",     &Camera::name)
    .def_readwrite ("is_ortho", &Camera::is_ortho)
    .def_readwrite ("zoom",     &Camera::zoom)
    .def_readwrite ("near",     &Camera::near)
    .def_readwrite ("far",      &Camera::far)
    .def_readwrite ("aperture", &Camera::aperture)
            
    .def ("__repr__", &Camera::repr,
            "x.__repr__() <==> repr(x)")
    ;       
}
