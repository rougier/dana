//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: link.cc 244 2007-07-19 10:09:40Z rougier $
// _____________________________________________________________________________

#include "link.h"
#include "unit.h"

using namespace dana::core;

// _________________________________________________________________________Link
Link::Link ()
{}

// _________________________________________________________________________Link
Link::Link (UnitPtr source, float weight)
     : source (source), weight (weight)
{}

// ________________________________________________________________________~Link
Link::~Link (void)
{}

// __________________________________________________________________________get
py::tuple const
Link::get (void)
{
    return py::make_tuple (source, weight);
}

// ___________________________________________________________________get_source
UnitPtr const
Link::get_source (void)
{
	return source;
}

// ___________________________________________________________________set_source
void
Link::set_source (UnitPtr source) 
{
    this->source = source;
}

// ___________________________________________________________________get_weight
float const
Link::get_weight (void)
{
	return weight;
}

// ___________________________________________________________________set_weight
void
Link::set_weight (float weight)
{
    this->weight = weight;
}

// ______________________________________________________________________compute
float
Link::compute (void)
{
    return source->potential * weight;
}

// ________________________________________________________________python_export
void
Link::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Link> >();
   
    class_<Link> ("Link",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A link describes the influence of a source over a target that owns the\n"
    "link.                                                                 \n"
    "______________________________________________________________________\n",

    init < UnitPtr, optional < float > > (
        (arg("source"),
         arg("weight") = 0.0f),
        "__init__ (source, weight=0)\n"))

    .add_property ("source",
                   &Link::get_source, &Link::set_source,
                   "source that feed the link")
    .add_property ("weight",
                   &Link::get_weight, &Link::set_weight,
                   "weight of the link")
     ;
}
