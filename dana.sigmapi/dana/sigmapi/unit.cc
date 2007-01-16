//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "core/map.h"
#include "core/layer.h"
#include "link.h"
#include "unit.h"
#include "spec.h"
#include <iostream>

using namespace boost::python::numeric;
using namespace dana::sigmapi;

// Constructor
// -----------------------------------------------------------------------------
Unit::Unit(void) : core::Unit()
{
	potential = 0.0;
}

// Destructor
// -----------------------------------------------------------------------------
Unit::~Unit(void)
{}

// Evaluate new potential and returns difference
// -----------------------------------------------------------------------------
float
Unit::evaluate (void)
{
    object spec = layer->map->get_spec();

    float tau      = extract<float> (spec.attr("tau"));
    float alpha    = extract<float> (spec.attr("alpha"));
    float baseline = extract<float> (spec.attr("baseline"));
    float min_act  = extract<float> (spec.attr("min_act"));
    float max_act  = extract<float> (spec.attr("max_act"));

    float input = 0;
    unsigned int size = afferents.size();

	for (unsigned int i=0; i<size; i++)
	{
		core::Link * aff = (afferents[i]).get();
		input += ((sigmapi::Link *)aff)->compute();
	}

	float lateral = 0;
   	size = laterals.size();

	for (unsigned int i=0; i<size; i++)
	{
		core::Link * lat = (laterals[i]).get();
        	lateral += ((sigmapi::Link*)lat)->compute();
	}
    	float du = (-potential + baseline + (1.0f/alpha)*(lateral + input)) / tau;
	float value = potential;
	potential += du;

	if (potential < min_act)
        potential = min_act;

	if (potential > max_act)
        potential = max_act;

	return value-potential;
}

void
Unit::connect(core::LinkPtr link)
{
	afferents.push_back(link);
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
