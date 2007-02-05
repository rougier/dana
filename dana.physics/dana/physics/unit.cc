//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "core/map.h"
#include "core/layer.h"
#include "core/link.h"
#include "unit.h"

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::physics;

// Constructor
// -----------------------------------------------------------------------------
Unit::Unit(void) : core::Unit()
{
    source = false;
}

// Destructor
// -----------------------------------------------------------------------------
Unit::~Unit(void)
{}

// Evaluate new potential and returns difference
// -----------------------------------------------------------------------------
float
Unit::compute_dp (void)
{
    if (!source) {
        potential = 0;
  	    for (unsigned int i=0; i<laterals.size(); i++)
            potential += laterals[i]->source->potential;
        potential /= laterals.size();
    }
	return potential;
}

// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Unit::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();
    
    class_<Unit, bases<core::Unit> >("Unit",
    "======================================================================\n"
    "\n"
    "A unit is a potential that is computed on the basis of some external\n"
    "sources that feed the unit. Those sources can be anything as long as\n"
    "they have some potential.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "   potential : unit potential (float)\n"
    "   source    : whether the unit is a source potential"
    "   position  : unit position within layer (tuple, read only)\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize unit\n")
        )
        
        .def_readwrite ("source", &Unit::source)
        ;
}
