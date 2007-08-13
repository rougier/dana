//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "robot.h"
#include "viewport.h"
#include "observer.h"

using namespace glpython::world::core;

//_________________________________________________________________________Robot
Robot::Robot (std::string name) : glpython::core::Object (name)
{
    h_camera = 0.5f;

	// The observer is initially positioned on the x axis, watching toward
    // the -x axis
    position[0] = 10.0f;
    position[1] = 0.0f;
    position[2] = 0.0f;

    up[0] = 0.0f;
    up[1] = 0.0f;
    up[2] = 1.0f;

    theta = 180;
    phi = 0;

    VectorsFromAngles();

    view = glpython::core::ViewportPtr (new Viewport());
    Observer * obs = dynamic_cast< world::core::Observer *>((view->observer).get());
    
    obs->allow_movement = false;

    obs->position[0] = position[0] + h_camera*up[0];
    obs->position[1] = position[1] + h_camera*up[1];
    obs->position[2] = position[2] + h_camera*up[2];

    obs->VectorsFromAngles();
    
}

//________________________________________________________________________~Robot
Robot::~Robot (void)
{}

//________________________________________________________________________render
void
Robot::render (void)
{
    float dx, dy, dz;
    dx = dy = dz = 1.0f;
     
    // We first draw the robot with its size
    robot(dx,dy,dz);
     
    // And finally the local axes for debuggin purposes
    local_axes();

    // And finally draw the observer
    Observer * obs = dynamic_cast< world::core::Observer *>((view->observer).get());    
    obs->render();
}

void Robot::local_axes(void)
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


//_________________________________________________________________________robot
void
Robot::robot (float dx, float dy, float dz)
{

    // Version cone
    GLUquadric* params;
    params = gluNewQuadric();

    glPolygonOffset (1,1);
    glEnable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glColor3f (1,0,0);

    glPushMatrix ();
    glTranslatef (position[0], position[1], position[2]);

    glRotatef (theta, 0, 0, 1);
    glRotatef (-phi, 0, 1, 0);

    gluCylinder(params,0.25,0,h_camera,20,1);

    glDisable (GL_POLYGON_OFFSET_FILL);
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glColor3f (0,0,0);

    gluCylinder(params,0.25,0,h_camera,20,1);

    glPopMatrix();

    gluDeleteQuadric(params);
}

//__________________________________________________________________________move
void
Robot::move (float dx, float dy, float dz) {
    position[0] += dx*forward[0]+dy*left[0]+dz*up[0]  ;
    position[1] += dx*forward[1]+dy*left[1]+dz*up[1];
    position[2] += dx*forward[2]+dy*left[2]+dz*up[2];

    VectorsFromAngles();

    Observer * obs = dynamic_cast< world::core::Observer *>((view->observer).get());

    obs->position[0] = position[0] + h_camera*up[0];
    obs->position[1] = position[1] + h_camera*up[1];
    obs->position[2] = position[2] + h_camera*up[2];

    obs->VectorsFromAngles();
}

//________________________________________________________________________rotate
void
Robot::rotate (float dtheta, float dphi) {
    theta += dtheta;
    phi += dphi;
    
    VectorsFromAngles();

    Observer * obs = dynamic_cast< world::core::Observer *>((view->observer).get());

    obs->position[0] = position[0] + h_camera*up[0];
    obs->position[1] = position[1] + h_camera*up[1];
    obs->position[2] = position[2] + h_camera*up[2];

    obs->rotate(dtheta,dphi);
}

//________________________________________________________________________rotate
void
Robot::rotateCamera (float dtheta, float dphi) {
    Observer * obs = dynamic_cast< world::core::Observer *>((view->observer).get());
    obs->rotate(dtheta,dphi);
}

//_____________________________________________________________VectorsFromAngles
void
Robot::VectorsFromAngles(void) {

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
    Util::cross_prod(forward,left,up);
    Util::normalize(up);
    look_at[0] = position[0] + forward[0];
    look_at[1] = position[1] + forward[1];
    look_at[2] = position[2] + forward[2];
}

//__________________________________________________________________________grab
void
Robot::grab (char * filename)
{
    Viewport * vport = dynamic_cast< world::core::Viewport *>(view.get());
    vport->save(filename);
}

//________________________________________________________________________append
void
Robot::append (glpython::core::ObjectPtr o)
{
    view->append(o);
}

//_________________________________________________________________python_export
void
Robot::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Robot> >();

    class_<Robot, bases <Object> >("Robot",
                                   init< optional<std::string> >("__init__ ()"))
        .def_readonly ("view", &Robot::view,"Viewport for the camera\n")
        .def("move",&Robot::move,"move(forward/backaward ,left/rigth ,up/down) : \n")
        .def("rotate",&Robot::rotate,"rotate(pan,tilt) : Rotate the robot and the camera\n")
        .def("rotateCamera",&Robot::rotateCamera,"rotateCamera(pan,tilt) : rotate the camera only\n")
        .def("append",&Robot::append,"Append(obj) append an object to the viewport\n")
        .def("grab",&Robot::grab,"Grab(filename) grab an image and save it to filename\n")
        ;       
}
