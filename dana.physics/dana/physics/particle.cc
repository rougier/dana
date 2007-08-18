//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#include "core/map.h"
#include "core/layer.h"
#include "core/link.h"
#include "particle.h"

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::physics;


// ============================================================================
//  constructor
// ============================================================================
Particle::Particle (void) : core::Unit()
{}

// ============================================================================
//  destructor
// ============================================================================
Particle::~Particle (void)
{}

// ============================================================================
//  evaluate new potential and returns difference
// ============================================================================
float
Particle::compute_dp (void)
{
    if ((rand()/float(RAND_MAX)) < .25) {
        int index = int ((rand()/float(RAND_MAX)) * laterals.size());
        float p = (laterals[index]->source->potential + potential)/2.0f;
        laterals[index]->source->potential = p;
        potential = p;
    }
	return 0.0f;
}

// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Particle::boost (void) {
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Particle> >();
    
    class_<Particle, bases<core::Unit> >("Particle",
    "======================================================================\n"
    "\n"
    "A particle is a potential that is computed on the basis of some"
    " external sources that feed the unit. Those sources can be anything as"
    " long as they have some potential.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "   potential : particle potential (float)\n"
    "   position  : particle position within layer (tuple, read only)\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize particle\n")
        )
        ;
}
