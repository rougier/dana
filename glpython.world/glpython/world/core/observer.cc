//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "observer.h"
#include "camera.h"
#include <GL/gl.h>

using namespace glpython::world::core;


//_____________________________________________________________________Observer
Observer::Observer (std::string name) : glpython::core::Observer(name)
{
    // Absolute position in space
    position[0] = 0;
    position[1] = 0;
    position[2] = 0;

    up[0] = 0;
    up[1] = 0;
    up[2] = 1;
    
    theta = 180;
    phi = 0;
    
    VectorsFromAngles();

    // We redefine the camera as a glpython::world::Camera
    camera = glpython::core::CameraPtr (new world::core::Camera());

    // Disable freefly property
    allow_movement = false;
}

//____________________________________________________________________~Observer
Observer::~Observer (void)
{}

//_________________________________________________________________________look
void
Observer::push (void)
{
    camera->push ();
    gluLookAt(position[0],position[1],position[2],look_at[0],look_at[1],look_at[2],up[0],up[1],up[2]);
}

void Observer::pop(void)
{
    camera->pop();
}

void Observer::render(void)
{
    float dx,dy,dz;
    dx = dy = dz = 0.25f;

    glPolygonOffset (1,1);
    glEnable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColor3f (0,1,0);

    glPushMatrix ();
    glTranslatef (position[0], position[1], position[2]);

    glRotatef (theta, 0, 0, 1);
    glRotatef (-phi, 0, 1, 0);

    glBegin (GL_QUADS);
    // Top Face (-z)
    glNormal3f (    0.0f,  0.0f,    1.0f);
    glVertex3f (    -dx/2.0f, -dy/2.0f,      dz/2.0);
    glVertex3f (      dx/2.0f,  -dy/2.0f,      dz/2.0);
    glVertex3f (      dx/2.0f,    dy/2.0f,      dz/2.0);
    glVertex3f (    -dy/2.0f,    dy/2.0f,      dz/2.0);
    // Bottom Face (-z)
    glNormal3f ( 0.0f,  0.0f, -1.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (-dx/2.0f, dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f, dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f,-dy/2.0f,-dz/2.0f);
    // Front face (x)
    glNormal3f ( 1.0f,  0.0f,  0.0f);
    glVertex3f (dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (dx/2.0f, dy/2.0f,-dz/2.0f);
    glVertex3f (dx/2.0f, dy/2.0f, dz/2.0f);
    glVertex3f (dx/2.0f,-dy/2.0f, dz/2.0f);
    // Rear Face (-x)
    glNormal3f (-1.0f,  0.0f,  0.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f, dz/2.0f);
    glVertex3f (-dx/2.0f, dy/2.0f, dz/2.0f);
    glVertex3f (-dx/2.0f, dy/2.0f,-dz/2.0f);
    // Left Face (y)
    glNormal3f ( 0.0f,  1.0f,  0.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f, dz/2.0f);
    glVertex3f ( dx/2.0f,-dy/2.0f, dz/2.0f);
    glVertex3f ( dx/2.0f,-dy/2.0f,-dz/2.0f);
    // Right Face (-y)
    glNormal3f ( 0.0f, -1.0f,  0.0f);
    glVertex3f (-dx/2.0f,dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f,dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f,dy/2.0f, dz/2.0f);
    glVertex3f (-dx/2.0f,dy/2.0f, dz/2.0f);
    glEnd();

    glDisable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glColor3f (0,0,0); 

    glBegin (GL_QUADS);
    // Top Face (-z)
    glNormal3f (    0.0f,  0.0f,    1.0f);
    glVertex3f (    -dx/2.0f, -dy/2.0f,      dz/2.0);
    glVertex3f (      dx/2.0f,  -dy/2.0f,      dz/2.0);
    glVertex3f (      dx/2.0f,    dy/2.0f,      dz/2.0);
    glVertex3f (    -dy/2.0f,    dy/2.0f,      dz/2.0);
    // Bottom Face (-z)
    glNormal3f ( 0.0f,  0.0f, -1.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (-dx/2.0f, dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f, dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f,-dy/2.0f,-dz/2.0f);
    // Front face (x)
    glNormal3f ( 1.0f,  0.0f,  0.0f);
    glVertex3f (dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (dx/2.0f, dy/2.0f,-dz/2.0f);
    glVertex3f (dx/2.0f, dy/2.0f, dz/2.0f);
    glVertex3f (dx/2.0f,-dy/2.0f, dz/2.0f);
    // Rear Face (-x)
    glNormal3f (-1.0f,  0.0f,  0.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f, dz/2.0f);
    glVertex3f (-dx/2.0f, dy/2.0f, dz/2.0f);
    glVertex3f (-dx/2.0f, dy/2.0f,-dz/2.0f);
    // Left Face (y)
    glNormal3f ( 0.0f,  1.0f,  0.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f,-dz/2.0f);
    glVertex3f (-dx/2.0f,-dy/2.0f, dz/2.0f);
    glVertex3f ( dx/2.0f,-dy/2.0f, dz/2.0f);
    glVertex3f ( dx/2.0f,-dy/2.0f,-dz/2.0f);
    // Right Face (-y)
    glNormal3f ( 0.0f, -1.0f,  0.0f);
    glVertex3f (-dx/2.0f,dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f,dy/2.0f,-dz/2.0f);
    glVertex3f ( dx/2.0f,dy/2.0f, dz/2.0f);
    glVertex3f (-dx/2.0f,dy/2.0f, dz/2.0f);
    glEnd();
    glPopMatrix ();

    local_axes();

}

//____________________________________________________________________local_axes
void
Observer::local_axes(void)
{
    glPushMatrix ();
    glTranslatef (position[0], position[1], position[2]);

    // Forward
    glColor3f(1,0,0);    
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f); // origin of the line
    glVertex3f(forward[0], forward[1], forward[2]); // ending point of the line
    glEnd( );

    // Right
    glColor3f(0,1,0);    
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f); // origin of the line
    glVertex3f(left[0], left[1], left[2]); // ending point of the line
    glEnd( );

    // up
    glColor3f(0,0,1);    
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f); // origin of the line
    glVertex3f(up[0], up[1], up[2]); // ending point of the line
    glEnd( );
    
    glPopMatrix ();
}

//________________________________________________________________________rotate
void
Observer::rotate (float dtheta, float dphi) {
    theta += dtheta;
    phi += dphi;
    
    VectorsFromAngles();
}

//_____________________________________________________________VectorsFromAngles
void
Observer::VectorsFromAngles(void) {
    if(phi > 89)
        phi = 89;
    else if (phi<-89)
        phi = -89;
    double r_temp = cos(phi*M_PI/180.0);
    forward[0] = r_temp * cos(theta*M_PI/180.0);
    forward[1] = r_temp * sin(theta*M_PI/180.0);
    forward[2] = sin(phi*M_PI/180.0);

    Util::cross_prod(up,forward,left);
    Util::normalize(left);
    //Util::cross_prod(forward,left,up);
    //Util::normalize(up);
    look_at[0] = position[0] + forward[0];
    look_at[1] = position[1] + forward[1];
    look_at[2] = position[2] + forward[2];
}

//_________________________________________________________pointer_motion_event
void
Observer::pointer_motion_event (float x, float y)
{
    if(button && allow_movement)
        {
            float dtheta = x;
            float dphi = y;
            rotate(-dtheta,dphi);
            VectorsFromAngles();
        }
    
}


//________________________________________________________________python_export
void
Observer::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Observer> >();

    class_<Observer, bases< glpython::core::Observer> >("Observer",
                                                        "======================================================================\n"
                                                        "                                                                      \n"
                                                        "           To be filled                                               \n"
                                                        "======================================================================\n",
                                                        
                                                        init< optional<std::string> >("__init__ ()"))
        .def("rotate",&Observer::rotate,"Rotate(dtheta,dphi)\n")
        
        .def_readwrite("allow_movement",&Observer::allow_movement,"Enable or disable freefly movable property\n")
        
        .def ("pointer_motion_event", &Observer::pointer_motion_event,
              "pointer_motion_event (x,y)")
        ;       
}
