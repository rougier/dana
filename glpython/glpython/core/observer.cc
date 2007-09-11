//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "observer.h"
#include <GL/gl.h>

using namespace glpython::core;


//_____________________________________________________________________Observer
Observer::Observer (std::string name) :
     name(name), phi(-45), theta(45)
{
    this->x = -1;
    this->y = -1;
    this->button = 0;
    camera = CameraPtr (new Camera());
}

//____________________________________________________________________~Observer
Observer::~Observer (void)
{}

//_________________________________________________________________________repr
std::string
Observer::repr (void)
{
    return name;
}

//_________________________________________________________________________look
void
Observer::push (void)
{
    camera->push ();
    glRotatef (-theta, 1, 0, 0);
    glRotatef (phi, 0, 0, 1);
}

//_________________________________________________________________________look
void
Observer::pop (void)
{
    camera->pop();
}

//_________________________________________________________________resize_event
void
Observer::resize_event (int x, int y, int w, int h)
{
    camera->w = w;
    camera->h = h;
}

//_________________________________________________________________select_event
void
Observer::select_event (int x, int y)
{
    camera->select_event (x, y);
}

//___________________________________________________________button_press_event
void
Observer::button_press_event (int button, int x, int y)
{
    this->x = x;
    this->y = y;
    this->button = button;
}

//_________________________________________________________button_release_event
void
Observer::button_release_event (int button, int x, int y)
{
    this->x = -1;
    this->y = -1;
    this->button = 0;
}

//_________________________________________________________pointer_motion_event
void
Observer::pointer_motion_event (int x, int y)
{
    if (this->button == 1) {
        if (x < 0)
            return;
        phi += (x - this->x) / 4.0;
        theta += (y - this->y) / 4.0;
        if (theta > 180.0)
            theta = 180.0;
        else if (theta < 0.0)
            theta = 0.0;
    } else if (button == 2) {
        camera->zoom += ((y-this->y)/float(camera->h))*3;
    }
    this->x = x;
    this->y = y;
}

//________________________________________________________________python_export
void
Observer::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Observer> >();

    class_<Observer>("Observer",
    "======================================================================\n"
    "                                                                      \n"
    " An observer is a point in space describing the position of a virtual \n"
    " observer looking at point (0,0,0) using a specific camera and can be \n"
    " moved using mouse.                                                   \n"
    "                                                                      \n"
    " Attributes:                                                          \n"
    "    camera - Camera this observer is using                            \n"
    "    phi - rotation around x axis                                      \n"
    "    theta - rotation aoound z axis (0 to pi)                          \n"
    "                                                                      \n"
    "======================================================================\n",
     
    init< optional<std::string> >("__init__ ()"))
        
    .def_readwrite ("name",   &Observer::name)
    .def_readwrite ("phi",    &Observer::phi)
    .def_readwrite ("theta",  &Observer::theta)
    .def_readwrite ("camera", &Observer::camera)

    .def ("__repr__", &Observer::repr,
          "x.__repr__() <==> repr(x)")

    .def ("resize_event", &Observer::resize_event,
          "resize_event (x,y,w,h)")

    .def ("button_press_event", &Observer::button_press_event,
          "button_press_event (button, x, y)")

    .def ("button_release_event", &Observer::button_release_event,
          "button_release_event (button,x,y)")

    .def ("pointer_motion_event", &Observer::pointer_motion_event,
          "pointer_motion_event (x,y)")
    ;       
}
