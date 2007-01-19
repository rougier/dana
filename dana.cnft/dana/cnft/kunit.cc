//
// Copyright (C) 2007,2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "core/map.h"
#include "core/layer.h"
#include "core/link.h"
#include "kunit.h"
#include "spec.h"

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::cnft;

// =============================================================================
//  constructor
// =============================================================================
KUnit::KUnit(void) : core::Unit()
{}

// =============================================================================
//  destructor
// =============================================================================
KUnit::~KUnit(void)
{}

// =============================================================================
//  computes potential and returns dp
// =============================================================================
float
KUnit::compute_dp (void)
{
    object spec = layer->map->get_spec();
    
    float tau      = extract<float> (spec.attr("tau"));
    float alpha    = extract<float> (spec.attr("alpha"));
    float baseline = extract<float> (spec.attr("baseline"));
    float min_act  = extract<float> (spec.attr("min_act"));
    float max_act  = extract<float> (spec.attr("max_act"));
    float wm = extract<float> (spec.attr("wm"));
    float wp = extract<float> (spec.attr("wp"));
    
	float input = 0;
    unsigned int size = afferents.size();

	for (unsigned int i=0; i<size; i++)
		input += afferents[i]->weight * afferents[i]->source->potential;

/*
	for (unsigned int i=0; i<size; i++)
		input += fabs(afferents[i]->weight - afferents[i]->source->potential);
    input = 3.5 * (1.0f - input/float(size));
*/

    float lateral = 0;
	float lateral_p = 0;
	float lateral_m = 0;
    size = laterals.size();

	for (unsigned int i=0; i<size; i++) {
        if ((laterals[i]->source->potential > 0) || (laterals[i]->weight > 0)) {
            float u = laterals[i]->weight * laterals[i]->source->potential;
            if (u >= 0)
                lateral_p += u;
            else
                lateral_m += u;
//            lateral += laterals[i]->weight * laterals[i]->source->potential;
        }
    }

    lateral = wp*lateral_p + wm*lateral_m;
    
    float du = (-potential + baseline + (1.0f/alpha)*(lateral + input)) / tau;
	float value = potential;
	potential += du;

	if (potential < min_act)
        potential = min_act;

	if (potential > max_act)
        potential = max_act;

	return value-potential;
}

// =============================================================================
//  computes weights and returns dw
// =============================================================================
float
KUnit::compute_dw (void)
{
    return 0.0f;

    if (potential < 0)
        return 0.0f;

    object spec = layer->map->get_spec();    
    float lrate = extract<float> (spec.attr("lrate"));
    
    lrate *= (1-potential);

    for (unsigned int i=0; i<afferents.size(); i++) {
        afferents[i]->weight = 
            (1-lrate)*afferents[i]->weight - 
            lrate*(afferents[i]->weight - afferents[i]->source->potential);
    }

    return 0.0f;
}

// ============================================================================
//    Boost wrapping code
// ============================================================================
void
KUnit::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<KUnit> >();
    
    class_<KUnit, bases<core::Unit> >("KUnit",
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
