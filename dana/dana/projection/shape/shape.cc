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
//
// =============================================================================
Shape::Shape (void)
{}

// =============================================================================
//
// =============================================================================
Shape::~Shape (void)
{}

// =============================================================================
//
// =============================================================================
int
Shape::call (std::vector<vec2i> &points,
             distance::DistancePtr distance,
             int x, int y, int w, int h)
{
    points.clear();
    return 0;
}

// =============================================================================
//
// =============================================================================
Point::Point (void) : Shape ()
{}

// =============================================================================
//
// =============================================================================
int
Point::call (std::vector<vec2i> &points,
             distance::DistancePtr distance,
             int x, int y, int w, int h)
{
    points.clear();
    points.push_back (vec2i (0,0));
    return 1;
}

// =============================================================================
//
// =============================================================================
Box::Box (float w, float h) : Shape (), width(w), height(h)
{}

// =============================================================================
//
// =============================================================================
int
Box::call (std::vector<vec2i> &points,
           distance::DistancePtr distance,
           int x, int y, int w, int h)
{
    points.clear();
    for (int i=-(w-1); i<w; i++) {
        for (int j=-(h-1); j<h; j++) {
            float dx = distance->call (x/float(w), 0, i/float(w), 0);
            float dy = distance->call (y/float(h), 0, j/float(h), 0);
            if ((dx <= width) && (dy <= height))
                points.push_back (vec2i(i,j));
        }
    }
    return points.size();
}

// =============================================================================
//
// =============================================================================
Disc::Disc (float r) : Shape (), radius(r)
{}

// =============================================================================
//
// =============================================================================
int
Disc::call (std::vector<vec2i> &points,
           distance::DistancePtr distance,
           int x, int y, int w, int h)
{
    points.clear();
    for (int i=-(w-1); i<w; i++) {
        for (int j=-(h-1); j<h; j++) {
            float d = distance->call (x/float(w), y/float(h), i/float(w), j/float(h));
            if (d <= radius)
                points.push_back (vec2i(i,j));
        }
    }
    return points.size();
}

// =============================================================================
//  Boost wrapping code
// =============================================================================
BOOST_PYTHON_MODULE(_shape)
{
    using namespace boost::python;

    class_<Shape> ("shape",
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
    
    class_<Point, bases <Shape> > ("point",
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

    class_<Box, bases <Shape> > ("box",
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
    
    class_<Disc, bases <Shape> > ("disc",
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


