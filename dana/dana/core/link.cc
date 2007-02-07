//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "link.h"
#include "unit.h"

using namespace dana::core;

//
// ----------------------------------------------------------------------------
Link::Link () /* Object(), */
{}

//
// ----------------------------------------------------------------------------
Link::Link (UnitPtr src, float w) : /* Object(), */ source(src), weight(w)
{}

//
// ----------------------------------------------------------------------------
Link::~Link (void)
{}

//
// ----------------------------------------------------------------------------
UnitPtr
Link::get_source (void) const
{
	return source;
}

//
// ----------------------------------------------------------------------------
void
Link::set_source (UnitPtr src) 
{
    source = src;
}

//
// ----------------------------------------------------------------------------
float
Link::get_weight (void) const
{
	return weight;
}

//
// ----------------------------------------------------------------------------
void
Link::set_weight (float w)
{
    weight = w;    
}

//
// ----------------------------------------------------------------------------
float
Link::compute()
{
    return source->potential * weight;
}


// ===================================================================
//  Boost wrapping code
// ===================================================================
void
Link::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Link> >();
   
    class_<Link> ("Link",
    "======================================================================\n"
    "\n"
    "A link describes the influence of a source over a target and is owned\n"
    "by the target.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "   source: source unit\n"
    "   weight: weight of the link\n"
    "\n"
    "======================================================================\n",
        
         init<UnitPtr, float> ()
         )
        .add_property ("weight", &Link::get_weight, &Link::set_weight)
        .add_property ("source", &Link::get_source, &Link::set_source)
        ;
}
