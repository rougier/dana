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
#include "munit.h"
#include "cnft/spec.h"
#include "core/spec.h"
#include <math.h>

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::learn;

// =============================================================================
//  constructor
// =============================================================================
MUnit::MUnit(void) : learn::Unit()
{
	max_layers.clear();
}

// =============================================================================
//  destructor
// =============================================================================
MUnit::~MUnit(void)
{}

// =============================================================================
//  Add a map in the set of maps used to compute the max operator in compute_dp
// =============================================================================
void
MUnit::add_maxLayer(core::Layer * layer)
{
	max_layers.push_back(layer);
}

// =============================================================================
//  computes potential and returns dp
// =============================================================================
float
MUnit::compute_dp (void)
{
    core::SpecPtr sp = layer->get_spec();
    if (sp.get() == NULL)
        return 0.0f;
    cnft::Spec *s = dynamic_cast<cnft::Spec *>(sp.get());
    if (s == 0)
        return 0.0f;
    
    float tau      = s->tau;
    float alpha    = s->alpha;
    float baseline = s->baseline;
    float min_act  = s->min_act;
    float max_act  = s->max_act;

    float input = 0;
    unsigned int size = afferents.size();

    // We compute a max over the afferent units that are in 
    // a layer of max_layers
    float max = 0.0;
    float temp_aff = 0.0;
	for (unsigned int i=0; i<size; i++)
        {
		if(!(std::find(max_layers.begin(),max_layers.end(),afferents[i]->get_source()->layer) == max_layers.end()))
		{
			// The source unit of the link afferents[i] is in max_layers 
			temp_aff = afferents[i]->compute() ;
			max  = (temp_aff > max ? temp_aff : max);
		} 
		else
		{
			input+= afferents[i]->compute();
		}
        }

    input += max;

	float lateral = 0;
	size = laterals.size();

	for (unsigned int i=0; i<size; i++)
        if ((laterals[i]->source->potential > 0) || (laterals[i]->weight > 0))
            lateral += laterals[i]->weight * laterals[i]->source->potential;

    float du = (-potential + baseline + (1.0f/alpha)*(lateral + input)) / tau;
	float value = potential;
	potential += du;

	if (potential < min_act)
        potential = min_act;

	if (potential > max_act)
        potential = max_act;

	return value-potential;
}


// ============================================================================
//    Boost wrapping code
// ============================================================================
void
MUnit::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<MUnit> >();
    
    class_<MUnit, bases<learn::Unit> >("MUnit",
    "======================================================================\n"
    "\n"
    "A MUnit extends learn::Unit that also extends cnft::Unit \n"
    " This unit computes a max over its afferents connections \n"
    " The other stuffs are unchangend\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize unit\n")
	)
	.def("add_maxLayer",&MUnit::add_maxLayer,"")
        ;
}
