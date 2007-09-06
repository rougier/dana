//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "projection.h"

using namespace dana;
using namespace dana::projection;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(connect_overloads, connect, 0, 1)

BOOST_PYTHON_MODULE(_projection) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Projection> >();
    
    class_<Projection, bases <core::Object> >
        ("Projection",
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
         init< optional <core::LayerPtr, core::LayerPtr, shape::ShapePtr,
                         profile::ProfilePtr, density::DensityPtr,
                         distance::DistancePtr, bool> > (

        (arg("src")          = core::LayerPtr(),
         arg("dst")          = core::LayerPtr(),
         arg("shape")        = shape::ShapePtr (new shape::Point()),
         arg("profile")      = profile::ProfilePtr (new profile::Constant (1.0f)),
         arg("density")      = density::DensityPtr (new density::Full()),
         arg("distance")     = distance::DistancePtr (new distance::Euclidean()),
         arg("self_connect") = false),

        
        "init() -- initializes the projection\n"
        )
    )

    .add_property ("shape",
                   &Projection::get_shape, &Projection::set_shape,
                   "Global shape")
    .add_property ("distance",
                   &Projection::get_distance, &Projection::set_distance,
                   "Distance to use to")                  
    .add_property ("density",
                   &Projection::get_density, &Projection::set_density,
                   "Connection probability function (of distance)")
    .add_property ("profile",
                   &Projection::get_profile, &Projection::set_profile,
                   "Weight profile function (of distance)")
   .add_property ("src",
                  &Projection::get_src, &Projection::set_src,
                  "Source layer (the one feeding)")
   .add_property ("dst",
                  &Projection::get_dst, &Projection::set_dst,
                  "Destination layer (the one receiving and owning the connection)")
    .def_readwrite ("self_connect", &Projection::self_connect) 
    
    .def ("connect", &Projection::connect,
          connect_overloads (args("data"), 
          "connect(data=None) -- instantiates the connection\n")
          )
    ;
}
