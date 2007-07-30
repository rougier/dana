//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "environment.h"

using namespace dana::cnft;

//__________________________________________________________________Environment
Environment::Environment (int number, float width, float intensity,
                          float radius, float theta, float dtheta, float noise)
    : core::Environment()

{
    this->number = number;
    this->width = width;
    this->intensity = intensity;
    this->radius = radius;
    this->theta = theta;
    this->dtheta = dtheta;
    this->noise = noise;
}

//_________________________________________________________________~Environment
Environment::~Environment (void)
{}

//_____________________________________________________________________evaluate
void
Environment::evaluate (void)
{
    core::MapPtr map = maps[0];
    if (map == core::MapPtr())
        return;    
    core::LayerPtr layer = map->get(0);
    if (layer == core::LayerPtr())
        return;

    float rx = map->width*radius*0.5f;
    float ry = map->height*radius*0.5f;
    float wx = map->width*width;
    float wy = map->height*width;

    layer->clear();
//    for (unsigned int i=0; i < layer->units.size(); i++)
//        layer->units[i]->potential = (rand()/float(RAND_MAX))*noise;

    for (int i=0; i< number; i++) {
        float t	= theta + i * (360.0f/number);
        float x = rx*cos (t/180.0f*M_PI) + map->width/2.0f;
        float y = ry*sin (t/180.0f*M_PI) + map->height/2.0f;
        gaussian (map, layer, x, y, wx, wy, intensity);
    }    
    theta = fmod (theta+dtheta, 360.0f);
}

//_____________________________________________________________________gaussian
void
Environment::gaussian (core::MapPtr map, core::LayerPtr layer,
                       float center_x, float center_y,
                       float width_x, float width_y,
                       float intensity)
{
	for (int j=0; j<map->height; j++) {
        for (int i=0; i<map->width; i++) {
            float v = intensity*exp (-(i-center_x)*(i-center_x)/(width_x*width_y)
                                     -(j-center_y)*(j-center_y)/(width_y*width_y));
            if (v > 1.0f)
                v = 1.0f;
            if (layer->get(i,j)->potential < v)
                layer->get(i,j)->potential = v;
        }
    }
}
 
// =============================================================================
//  python export
// =============================================================================
void
Environment::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Environment> >();

    class_<Environment, bases <core::Environment> > ("Environment",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "This environment represents gaussian rotating gaussian stimuli        \n"
    "                                                                      \n"
    "Attributes                                                            \n"
    "----------                                                            \n"
    "  number -- Number of stimulus                                        \n"
    "  intensity -- Stimulus intensity                                     \n"
    "  width --  Stimulus width                                            \n"
    "  radius -- Stimulus distance from map center                         \n"
    "  theta -- Current rotation                                           \n"
    "  dtheta -- Rotation speed                                            \n"    
    "  noise -- Overall noise level                                        \n"    
    "______________________________________________________________________\n",
    init < optional <int, float, float, float, float, float, float> > (
        (arg("number") = 3,
         arg("width") = 0.1f,
         arg("intensity") = 1.5f,
         arg("radius") = 0.7f,
         arg("theta") = 0.0f,
         arg("dtheta") = 1.0f,
         arg("noise") = 0.0f),
        "__init__ (number, width, intensity, radius, theta, dtheta, noise)\n"))

        .def_readwrite ("number",    &Environment::number)
        .def_readwrite ("width",     &Environment::width)
        .def_readwrite ("intensity", &Environment::intensity)
        .def_readwrite ("radius",    &Environment::radius)
        .def_readwrite ("theta",     &Environment::theta)
        .def_readwrite ("dtheta",    &Environment::dtheta)
        .def_readwrite ("noise",     &Environment::noise)
        
        .def ("attach", &Environment::attach,
        "attach(map) -- Attach a map to the environment\n")

        .def ("evaluate", &Environment::evaluate,
        "evaluate() -- Evaluate environment state\n")
        ;
}
