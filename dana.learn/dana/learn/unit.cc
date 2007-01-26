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
#include "unit.h"
#include "cnft/spec.h"

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::learn;

// =============================================================================
//  constructor
// =============================================================================
Unit::Unit(void) : cnft::Unit()
{}

// =============================================================================
//  destructor
// =============================================================================
Unit::~Unit(void)
{}

// =============================================================================
//  Define the learning rule
// =============================================================================
 
void Unit::set_learning_rule(std::vector<float> learnFunc)
{
	this->learnFunc = learnFunc;
}

// =============================================================================
//  Learn
// =============================================================================

void Unit::learn(core::LayerPtr dst,float scale)
{
	printf("learn ! \n");
	if(dst.get() == layer)
	{
		printf("test ! \n");
		// Learn the lateral links
		float dw = 0.0;
		for(int i = 0 ; laterals.size() ; i++)
		{
			printf("learn2 ! \n");
			float vi = potential;
			float vj = laterals[i]->get_source()->potential;
			float w = laterals[i]->get_weight();
		}
	}
	else
	{
		// Learn the afferent links
	}
}

// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Unit::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();
    
    class_<Unit, bases<cnft::Unit> >("Unit",
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
	.def ("set_lrule", &Unit::set_learning_rule,
	      "set_lrule(int[]) -- defines the learning rule\n")
	.def("learn",&Unit::learn,
	     "learn() -- Learns the weights with the defined learning rule\n")
        ;
}
