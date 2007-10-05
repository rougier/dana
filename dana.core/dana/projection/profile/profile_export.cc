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

BOOST_PYTHON_MODULE(_profile)
{
    using namespace boost::python;
    
    register_ptr_to_python< boost::shared_ptr<Profile> >();
    register_ptr_to_python< boost::shared_ptr<Constant> >();
    register_ptr_to_python< boost::shared_ptr<Linear> >();
    register_ptr_to_python< boost::shared_ptr<Uniform> >();  
    register_ptr_to_python< boost::shared_ptr<Gaussian> >();
    register_ptr_to_python< boost::shared_ptr<DoG> >();
    docstring_options doc_options;
    doc_options.disable_signatures();

    
    class_<Profile>("Profile",
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
    
    class_<Constant, bases <Profile> >("Constant",
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
    
     class_<Linear, bases <Profile> >("Linear",
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
    
    class_<Uniform, bases <Profile> >("Uniform",
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
    
    class_<Gaussian, bases <Profile> >("Gaussian",
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
    
    class_<DoG, bases <Profile> >("DoG",
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
