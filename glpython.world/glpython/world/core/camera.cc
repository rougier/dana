//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
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

using namespace glpython::world::core;


//_______________________________________________________________________Camera
Camera::Camera (std::string name) : glpython::core::Camera(name)
{
    near = 0.00001;
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
}

//________________________________________________________________python_export
void
Camera::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Camera> >();

    class_<Camera, bases< glpython::core::Camera> >("Camera",
                                                    "======================================================================\n"
                                                    "                  TO BE FILLED                                        \n"
                                                    "======================================================================\n",
                                                    
                                                    init< optional<std::string> >("__init__ ()"))
        
        ;       
}
