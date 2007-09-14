//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#define GL_GLEXT_PROTOTYPES

#include <GL/glu.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <iostream>
#include "viewport.h"

#include <fstream>

using namespace glpython::world::core;


//_____________________________________________________________________Viewport
Viewport::Viewport (tuple size, tuple position,
                    bool has_border,
                    bool is_ortho,
                    std::string name) : glpython::core::Viewport (size,position,has_border,is_ortho,name)
{
    observer = glpython::core::ObserverPtr (new Observer("Freefly camera"));
    Observer * obs = dynamic_cast< world::core::Observer *>(observer.get());
    obs->position[0] = 10.0f;
    obs->position[1] = 0.0f;
    obs->position[2] = 0.0f;
    obs->theta = 180;
    obs->phi = 0;
    obs->VectorsFromAngles();

    obs->allow_movement = true;

    button_pressed = false;
}

//____________________________________________________________________~Viewport
Viewport::~Viewport (void)
{
}

//_________________________________________________________________________save

void Viewport::save(char * filename, int save_width, int save_height)
{}

//______________________________________________________________key_press_event
void
Viewport::key_press_event (std::string key)
{
    if ((!visible) || (!has_focus))
        return;
    Observer * obs = dynamic_cast< world::core::Observer *>(observer.get());
    if(obs->allow_movement)
        {
            
            if(key == "up")
                {
                    obs->position[0] += 0.1*obs->forward[0];
                    obs->position[1] += 0.1*obs->forward[1];
                    obs->position[2] += 0.1*obs->forward[2];

                    obs->look_at[0] += 0.1*obs->forward[0];
                    obs->look_at[1] += 0.1*obs->forward[1];
                    obs->look_at[2] += 0.1*obs->forward[2];
                }
            else if(key == "down")
                {
                    obs->position[0] -= 0.1*obs->forward[0];
                    obs->position[1] -= 0.1*obs->forward[1];
                    obs->position[2] -= 0.1*obs->forward[2];

                    obs->look_at[0] -= 0.1*obs->forward[0];
                    obs->look_at[1] -= 0.1*obs->forward[1];
                    obs->look_at[2] -= 0.1*obs->forward[2];
                }
            else if(key == "left")
                {
                    obs->position[0] += 0.1*obs->left[0];
                    obs->position[1] += 0.1*obs->left[1];
                    obs->position[2] += 0.1*obs->left[2];

                    obs->look_at[0] += 0.1*obs->left[0];
                    obs->look_at[1] += 0.1*obs->left[1];
                    obs->look_at[2] += 0.1*obs->left[2];
                }
            else if(key == "right")
                {
                    obs->position[0] -= 0.1*obs->left[0];
                    obs->position[1] -= 0.1*obs->left[1];
                    obs->position[2] -= 0.1*obs->left[2];

                    obs->look_at[0] -= 0.1*obs->left[0];
                    obs->look_at[1] -= 0.1*obs->left[1];
                    obs->look_at[2] -= 0.1*obs->left[2];            
                }
            else if(key == "home")
                {
                    obs->position[0] += 0.1*obs->up[0];
                    obs->position[1] += 0.1*obs->up[1];
                    obs->position[2] += 0.1*obs->up[2];

                    obs->look_at[0] += 0.1*obs->up[0];
                    obs->look_at[1] += 0.1*obs->up[1];
                    obs->look_at[2] += 0.1*obs->up[2];            
                }
            else if(key == "end")
                {
                    obs->position[0] -= 0.1*obs->up[0];
                    obs->position[1] -= 0.1*obs->up[1];
                    obs->position[2] -= 0.1*obs->up[2];

                    obs->look_at[0] -= 0.1*obs->up[0];
                    obs->look_at[1] -= 0.1*obs->up[1];
                    obs->look_at[2] -= 0.1*obs->up[2];            
                }
            else if(key == "g")
                {
                    std::cout << "Viewport geometry : " 
                              << " [ " << geometry[0]
                              << " ; " << geometry[1]
                              << " ; " << geometry[2]
                              << " ; " << geometry[3]
                              << " ] " << std::endl;
                }
            // else do nothing
        }
    
}

//_________________________________________________________pointer_motion_event
void
Viewport::pointer_motion_event (int x, int y)
{
    if ((!visible) || (!has_focus))
        return;
    if (!child_has_focus) {
        dx = (x-(geometry[0]+geometry[2]/2.0))/geometry[2];
        dy = (y - (geometry[1]+geometry[3]/2.0))/geometry[3];

    } else {
        for (unsigned int i=0; i<viewports.size(); i++) {
            if (viewports[i]->has_focus)
                viewports[i]->pointer_motion_event (x, y);
        }
    }
}

//_______________________________________________________________________render
void
Viewport::render ()
{
    if(button_pressed)
        {
            Observer * obs = dynamic_cast< world::core::Observer *>(observer.get());
            if(obs->allow_movement)
                {
                    obs->pointer_motion_event (dx, dy);
                }
        }
    glpython::core::Viewport::render();
}

void
Viewport::button_press_event (int button, int x, int y)
{
    glpython::core::Viewport::button_press_event(button,x,y);
    dx = (x-(geometry[0]+geometry[2]/2.0))/geometry[2];
    dy = (y - (geometry[1]+geometry[3]/2.0))/geometry[3];
    button_pressed = true;
}

void
Viewport::button_release_event (int button, int x, int y)
{
    glpython::core::Viewport::button_release_event(button,x,y);
    button_pressed = false;
}

//________________________________________________________________python_export
void
Viewport::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Viewport> >();

    class_<ViewportWrapper, bases<glpython::core::Viewport>,boost::noncopyable  > (
        "Viewport",
        "======================================================================\n"
        " A world::Viewport inherits core::Viewport and provides a method      \n"
        " that permits to freefly in the scene                                 \n"
        "======================================================================\n",
        init< optional < tuple, tuple, bool, bool, std::string > > (
            (arg("size") = make_tuple (1.0f,1.0f),
             arg("position") = make_tuple (0.0f,0.0f),
             arg("has_border") = true,
             arg("is_ortho") = false,
             arg("name") = "Viewport"),
            "__init__ (size, position, has_border, name )\n"))
        
        .def ("key_press_event", &Viewport::key_press_event,
              "key_press_event (key)\n")
        
        .def ("button_press_event", &Viewport::button_press_event,
              "button_press_event (button, x, y)")
        
        .def ("button_release_event", &Viewport::button_release_event,
              "button_release_event (button,x,y)")
        
        .def ("pointer_motion_event", &Viewport::pointer_motion_event,
              "pointer_motion_event (x,y)\n")
        
        .def("save", &Viewport::save, &ViewportWrapper::default_save,"save(filename) : save a snapshot of the viewport in filename\n")
        ;       
}
