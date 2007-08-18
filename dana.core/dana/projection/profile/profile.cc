//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "profile.h"

using namespace dana::projection;
using namespace dana::projection::profile;


// =============================================================================
//
// =============================================================================
Profile::Profile (void)
{}

// =============================================================================
//
// =============================================================================
Profile::~Profile (void)
{}

// =============================================================================
//
// =============================================================================
float
Profile::call (float distance)
{
    return 0.0f;
}

// =============================================================================
//
// =============================================================================
Constant::Constant (float v) : Profile()
{
    value = v;
}

// =============================================================================
//
// =============================================================================
float
Constant::call (float distance)
{
    return value;
}

// =============================================================================
//
// =============================================================================
Linear::Linear (float min, float max) : Profile()
{
    minimum = min;
    maximum = max;
}

// =============================================================================
//
// =============================================================================
float
Linear::call (float distance)
{
    return minimum + (maximum - minimum)*(1.0f-distance);
}

// =============================================================================
//
// =============================================================================
Uniform::Uniform (float min, float max) : Profile()
{
    minimum = min;
    maximum = max;
}

// =============================================================================
//
// =============================================================================
float
Uniform::call (float distance)
{
    return minimum + (maximum - minimum)*drand48();
}

// =============================================================================
//
// =============================================================================
Gaussian::Gaussian (float s, float m) : Profile()
{
    scale = s;
    mean = m * m;
}

// =============================================================================
//
// =============================================================================
float
Gaussian::call (float distance)
{
    return scale * exp (-(distance*distance)/ mean);
}

// =============================================================================
//
// =============================================================================
DoG::DoG (float s1, float m1, float s2, float m2) : Profile()
{
    scale_1 = s1;
    mean_1  = m1 * m1;
    scale_2 = s2;
    mean_2  = m2 * m2;    
}

// =============================================================================
//
// =============================================================================
float
DoG::call (float distance)
{
    float d = distance*distance;
    return scale_1 * exp (-d/mean_1) - scale_2 * exp (-d/mean_2);
}


// =============================================================================
//    Boost wrapping code
// =============================================================================
BOOST_PYTHON_MODULE(_profile)
{
    using namespace boost::python;
    
    register_ptr_to_python< boost::shared_ptr<Profile> >();
    register_ptr_to_python< boost::shared_ptr<Constant> >();
    register_ptr_to_python< boost::shared_ptr<Linear> >();
    register_ptr_to_python< boost::shared_ptr<Uniform> >();  
    register_ptr_to_python< boost::shared_ptr<Gaussian> >();
    register_ptr_to_python< boost::shared_ptr<DoG> >();
    
    
    class_<Profile>("profile",
    "======================================================================\n"
    "\n"
    "A profile is a weight function that depends on the distance between a\n"
    "source and a target.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< > (
        "__init__() -- initializes profile\n")
        )
    
        .def ("__call__", &Profile::call,
        "__call__(d) -> return weight for distance d\n")
    ;
    
    class_<Constant, bases <Profile> >("constant",
    "======================================================================\n"
    "\n"
    "The constant profile gives a constant connection weight between any\n"
    "source and any target.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< float > (
        "__init__(value) -- initializes profile\n")
        )
    ;
    
     class_<Linear, bases <Profile> >("linear",
    "======================================================================\n"
    "\n"
    "The linear profile varies from a minimum at distance 1 to a maximum at\n"
    "distance 0.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< float,float > (
        "__init__(min,max) -- initializes profile\n")
        )
    ;
    
    class_<Uniform, bases <Profile> >("uniform",
    "======================================================================\n"
    "\n"
    "The uniform profile gives random weight between a minimum and a maximum\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< float,float > (
        "__init__(min,max) -- initializes profile\n")
        )
    ;
    
    class_<Gaussian, bases <Profile> >("gaussian",
    "======================================================================\n"
    "\n"
    "The gaussian profile gives gaussian weight as a function of distance \n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< float,float > (
        "__init__(scale,mean) -- initializes profile\n")
        )
    ;
    
    class_<DoG, bases <Profile> >("dog",
    "======================================================================\n"
    "\n"
    "The DoG profile gives difference of gaussians as a function of distance\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< float,float,float,float > (
        "__init__(scale1,mean1,scale2,mean2) -- initializes profile\n")
        )
    ;
}
