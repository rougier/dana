//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$


#include "link.h"
#include "unit.h"

using namespace dana::sigmapi;
using namespace dana;

//
// ----------------------------------------------------------------------------
Link::Link (LinkType t) : core::Link()
{
    type = t;
}

//
// ----------------------------------------------------------------------------
Link::~Link (void)
{}

//
// ----------------------------------------------------------------------------
core::UnitPtr
Link::get_source (const int i) const
{
    return source[i];
}

//
// ----------------------------------------------------------------------------
void
Link::add_source (core::UnitPtr src)
{
    source.push_back(src);
}

//
// ----------------------------------------------------------------------------
float
Link::compute(void)
{
    float value = 0.0;
    switch(type)
    {
    case SIGMAPI_MAX:
        {
            for(int i = 0 ; i < source.size() ; i ++)
            {
                value = MAX(value, source[i]->potential);
            }
            break;
        }
    case SIGMAPI_PROD:
        {
            value = 1.0;
            for(int i = 0 ; i < source.size(); i++)
            {
                value *= source[i]->potential;
            }
            value*= weight;
            break;
        }
    default:
        break;
    }
    return value;
}


// ===================================================================
//  Boost wrapping code
// ===================================================================
void
Link::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Link> >();

    class_<Link, bases<core::Link> > ("Link",
                                      "======================================================================\n"
                                      "\n"
                                      "A link describes the influence of a source over a target and is owned\n"
                                      "by the target.\n"
                                      "\n"
                                      "Attributes:\n"
                                      "-----------\n"
                                      "   source: source units\n"
                                      "   weight: weight of the link\n"
                                      "\n"
                                      "======================================================================\n",

                                      init<LinkType> ()
                                     )
    .add_property ("weight", &Link::get_weight, &Link::set_weight)
    ;
}
