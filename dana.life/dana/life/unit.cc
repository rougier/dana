//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "dana/core/map.h"
#include "dana/core/layer.h"
#include "dana/core/link.h"
#include "unit.h"

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::life;

// ============================================================================
//  constructor
// ============================================================================
Unit::Unit(void) : core::Unit()
{
}

// ============================================================================
//  destructor
// ============================================================================
Unit::~Unit(void)
{}

// ============================================================================
//  computes potential
// ============================================================================
float
Unit::compute_dp (void)
{
    float n= 0;
    for (unsigned int i=0; i<laterals.size(); i++)
        n += laterals[i]->source->potential;
    
    if ((potential == 0) && (n == 3))
        _potential = 1.0;
    else if ((n < 2) || (n > 3))
        _potential = 0.0f;

    potential = _potential;

	return 0;
}

// ============================================================================
//  computes weights
// ============================================================================
float
Unit::compute_dw (void)
{
    potential = _potential;
	return 0;
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
    "   position  : unit position within layer (tuple, read only)\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize unit\n")
        )
        ;
}
