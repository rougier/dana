//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: link.cc 244 2007-07-19 10:09:40Z rougier $

#include "link.h"
#include "unit.h"

using namespace dana::core;

//_________________________________________________________________________Link
Link::Link ()
{}

//_________________________________________________________________________Link
Link::Link (UnitPtr source, float weight)
     : source (source), weight (weight)
{}

//________________________________________________________________________~Link
Link::~Link (void)
{}

//_______________________________________________________________get/set source
UnitPtr
Link::get_source (void)
{
	return source;
}

void
Link::set_source (UnitPtr source) 
{
    this->source = source;
}

//_______________________________________________________________get/set weight
float
Link::get_weight (void)
{
	return weight;
}

void
Link::set_weight (float weight)
{
    this->weight = weight;
}

//______________________________________________________________________compute
float
Link::compute (void)
{
    return source->potential * weight;
}

//________________________________________________________________python_export
void
Link::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Link> >();
   
    class_<Link> ("Link",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A link describes the influence of a source over a target that owns    \n"
    "the link.                                                             \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "   source -- source unit                                              \n"
    "   weight -- weight of the link                                       \n"
    "______________________________________________________________________\n",

    init < UnitPtr, optional < float > > (
        (arg("source"),
         arg("weight") = 0.0f),
        "__init__ (source, weight=0)\n"))
    .add_property ("source", &Link::get_source, &Link::set_source)        
    .add_property ("weight", &Link::get_weight, &Link::set_weight)
    ;
}
