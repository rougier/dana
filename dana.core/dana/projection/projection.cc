//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <iostream>
#include "projection.h"
#include "projector.h"
#include "../core/map.h"
#include "../core/unit.h"
#include "../core/link.h"

using namespace dana::projection;

// ___________________________________________________________________Projection
Projection::Projection (void) : Object ()
{
    self_connect = true;
}

// __________________________________________________________________~Projection
Projection::~Projection (void)
{}

// ______________________________________________________________________connect
void
Projection::connect (object data)
{
    int src_width  = src->map->width;
    int src_height = src->map->height;
    int dst_width  = dst->map->width;
    int dst_height = dst->map->height;

    // Relative shape from a unit
    std::vector<shape::vec2i> points;
    shape->call (points, distance, 0, 0, src_width, src_height);
    
    for (int i=0; i<dst->size(); i++) {
        int dst_x = (i % dst_width);
        int dst_y = (i / dst_width);
        float x0 = dst_x/float(dst_width);
        float y0 = dst_y/float(dst_height);
        //int src_x = int (dst_x * (float(src_width)/float(dst_width)));
        //int src_y = int (dst_y * (float(src_height)/float(dst_height)));
        //float cd = (*distance) (x0,y0,.5f,.5f);
        int src_x = int (dst_x * src_width/dst_width);
        int src_y = int (dst_y * src_height/dst_height);        
        for (unsigned int j=0; j<points.size(); j++) {
            int x = src_x - points[j].x;
            int y = src_y - points[j].y;
            if ((x >= 0) && (y>=0) && (x<src_width) && (y<src_height)) {
                float x1 = x/float(src_width);
                float y1 = y/float(src_height);
                float d = distance->call (x0,y0,x1,y1);
                float de = density->call (d);
                if ((de) && (self_connect || (dst->get(i) != src->get (y*src_width +x)))) {
                    float w = profile->call(d);
                    dst->get(i)->connect (src->get (y*src_width+x), w, data);
                }
            }
        }
    }    
}


// ________________________________________________________________python_export
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(connect_overloads, connect, 0, 1)

void
Projection::python_export (void) {
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Projection> >();

    class_<Projection, bases <core::Object> > ("Projection",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A projection is the specification of a pattern of connection between  \n"
    "two layers. It can be precisely defined using four different notions: \n"
    "                                                                      \n"
    " distance : it defines how to measure distances between a source and  \n"
    "            a target and can be either the euclidean, the manhattan   \n"
    "            or the max distance. Each distance can be made toric.     \n"
    "                                                                      \n"
    " shape    : it defines the most general set of sources that can       \n"
    "            potentially be connected to a target. It can be a point,  \n"
    "            a box of a given size or a disc of a given radius.        \n"
    "                                                                      \n"                       
    " profile  : it defines connection weights as a function of the        \n"
    "            distance between a source and a target.                   \n"
    "                                                                      \n"
    " density  : it defines the probability of a connection to be actually \n"
    "            instantiated as a function of the distance.               \n"
    "______________________________________________________________________\n",
    init<> (
        "init() -- initializes the projection\n"
        )
    )
    
    .def_readwrite ("self_connect", &Projection::self_connect) 
    .def_readwrite ("src", &Projection::src)
    .def_readwrite ("dst", &Projection::dst)
    .def_readwrite ("shape", &Projection::shape)
    .def_readwrite ("distance", &Projection::distance)
    .def_readwrite ("density", &Projection::density)
    .def_readwrite ("profile", &Projection::profile)
    
    .def ("connect", &Projection::connect,
          connect_overloads (args("data"), 
          "connect(data=None) -- instantiates the connection\n")
          )
    ;
}

BOOST_PYTHON_MODULE(_projection)
{
    Projection::python_export();
}
