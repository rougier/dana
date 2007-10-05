//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "shape.h"

using namespace dana::projection;
using namespace dana::projection::shape;


// =============================================================================
//  Boost wrapping code
// =============================================================================
BOOST_PYTHON_MODULE(_shape)
{
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Shape> >();
    register_ptr_to_python< boost::shared_ptr<Point> >();
    register_ptr_to_python< boost::shared_ptr<Box> >();
    register_ptr_to_python< boost::shared_ptr<Disc> >();    
    docstring_options doc_options;
    doc_options.disable_signatures();

    
    class_<Shape> ("Shape",
    "======================================================================\n"
    "\n"
    "Generic shape object\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< > (
        "__init__() -- initializes shape\n")
        )
    ;
    
    class_<Point, bases <Shape> > ("Point",
    "======================================================================\n"
    "\n"
    "Point shape\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< > (
        "__init__() -- initializes shape\n")
        )
    ;

    class_<Box, bases <Shape> > ("Box",
    "======================================================================\n"
    "\n"
    "Box shape of a given width and height\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< float,float > (
        "__init__(w,h) -- initializes shape\n")
        )
    ;
    
    class_<Disc, bases <Shape> > ("Disc",
    "======================================================================\n"
    "\n"
    "Disc shape of a given radius\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init< float > (
        "__init__(r) -- initializes shape\n")
        )
    ;
}


