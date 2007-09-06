//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <cstdlib>
#include "density.h"

using namespace dana::projection;
using namespace dana::projection::density;

// =============================================================================
//    Boost wrapping code
// =============================================================================
BOOST_PYTHON_MODULE(_density)
{
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Density> >();
    register_ptr_to_python< boost::shared_ptr<Full> >();
    register_ptr_to_python< boost::shared_ptr<Sparse> >();
    register_ptr_to_python< boost::shared_ptr<Sparser> >();    

    class_<Density>("Density",
    "======================================================================\n"
    "\n"
    "A density represents the probability of a connection to be made\n"
    "according to the distance between the source and the target.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< optional<float> > (
        "__init__() -- initializes density\n")
        )
    
        .def ("__call__", &Density::call,
        "__call__(d) -> return true or false depending on d\n")
    ;
    
    class_<Full, bases <Density> >("Full",
    "======================================================================\n"
    "\n"
    "The full density represents a probability of 1 to connect a source to a\n"
    "target.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< optional<float> > (
        "__init__() -- initializes density\n")
        )
    ;
    
    class_<Sparse, bases <Density> >("Sparse",
    "======================================================================\n"
    "\n"
    "The sparse density represents a uniform probability to connect a source\n"
    "to a target.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< optional<float> > (
        "__init__() -- initializes density\n")
        )
    ;

    class_<Sparser, bases <Density> >("Sparser",
    "======================================================================\n"
    "\n"
    "The sparser density represents a normal-like probability to connect a\n"
    "source to a target.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< optional<float> > (
        "__init__() -- initializes density\n")
        )
    ;
    
}

