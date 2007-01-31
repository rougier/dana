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
#include <math.h>

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
 
void Unit::set_learning_rule(std::vector<std::vector<float> > * learnFunc)
{
	this->learnFunc = learnFunc;
}

// =============================================================================
//  Learn
// =============================================================================

void Unit::learn(core::LayerPtr src,float scale)
{
	if(src.get() == layer)
	{
		// Learn the lateral links
		float dw = 0.0;
		float value = 0.0;
		float vi,vj,w;
		int puissi,puissj = 0;
		std::vector<std::vector<float> > learnF = *learnFunc;
		for(int i = 0 ; i < laterals.size() ; i++)
		{
			dw = 0.0;
			vi = potential;
			vj = laterals[i]->get_source()->potential;
			w = laterals[i]->get_weight();
			for(int j = 0 ; j < learnF.size() ; j++)
			{
				value = 0.0;
				puissi = int(learnF[j][0]);
				puissj = int(learnF[j][1]);
				for( int k = 2 ; k < learnF[j].size() ; k++)
				{
					value += learnF[j][k]*pow(w,k-2);
				}
				value *= pow(vi,puissi)*pow(vj,puissj);
				dw += value;
			}
			dw *= scale;	
			laterals[i]->set_weight(w+dw);	
		}
	}
	else
	{
		// Learn the afferent links
		float dw = 0.0;
		float value = 0.0;
		float vi,vj,w;
		int puissi,puissj = 0;
		std::vector<std::vector<float> > learnF = *learnFunc;
		for(int i = 0 ; i < afferents.size() ; i++)
		{
			if(afferents[i]->get_source()->layer == src.get())
			{
				dw = 0.0;
				vi = potential;
				vj = afferents[i]->get_source()->potential;
				w = afferents[i]->get_weight();
				for(int j = 0 ; j < learnF.size() ; j++)
				{
					value = 0.0;
					puissi = int(learnF[j][0]);
					puissj = int(learnF[j][1]);
					for( int k = 2 ; k < learnF[j].size() ; k++)
					{
						value += learnF[j][k]*pow(w,k-2);
					}
					value *= pow(vi,puissi)*pow(vj,puissj);
					dw += value;
				}
				dw *= scale;	
				afferents[i]->set_weight(w+dw);	
			}
		}
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
    "Attributes (inherited from cnft.Unit):\n"
    "-----------\n"
    "   potential : unit potential (float) \n"
    "   position  : unit position within layer (tuple, read only)\n"
    "\n"
    "Methods\n"
    "----------\n"
    "   set_lrule : defines the learning rule to use\n"
    "   learn     : learn with a given learning rate\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize unit\n")
	)
	.def ("set_lrule", &Unit::set_learning_rule,
	      "set_lrule(float[][]) -- defines the learning rule. See Learner's documentation for details\n")
	.def("learn",&Unit::learn,
	     "learn(Layer src,float lrate) -- Learns the weights from src, with learning rate lrate, with the defined learning rule\n")
        ;
}
