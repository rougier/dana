//
// Copyright (C) 2007,2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "unit.h"
#include "learner.h"
#include <iostream>

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::learn;

// =============================================================================
//  constructor
// =============================================================================
Learner::Learner(void)
{}

// =============================================================================
//  destructor
// =============================================================================
Learner::~Learner(void)
{}

// =============================================================================
//  Define the learning rule
// =============================================================================
void Learner::add(core::LayerPtr src,core::LayerPtr dst,boost::python::numeric::array params)
{
	printf("salut");
	if(!PyArray_Check(params.ptr())){
		PyErr_SetString(PyExc_ValueError, "expected a PyArrayObject");
		throw_error_already_set();
	}
	double * dataPtr = (double*)PyArray_DATA(params.ptr());
	int size = PyArray_Size(params.ptr());
	
	std::vector<float> learn_params;
	
	for(int i = 0 ; i < size ; i++)
	{
		learn_params.push_back(*(dataPtr + i ));
 		
	}
	learnStr learn;
	learn.source = src;
	learn.destination = dst;
	//printf("Dest size : %i\n",dst->size());
	learn.params = learn_params;
	learns.push_back(learn);
}

// =============================================================================
//  Learn
// =============================================================================

void Learner::learn(float scale)
{
	learnStr learn;
	core::LayerPtr src,dst;
	learn::Unit * dst_unit;
	for(int i = 0 ; i < learns.size() ; i++)
	{
		learn = learns[i];
		src = learn.source;
		dst = learn.destination;
		for(int j = 0 ; j < dst->size() ; j++)
		{
			dst_unit = (learn::Unit*)((dst->get(j)).get());
			dst_unit->set_learning_rule(&(learn.params));
			//printf("Learning of UNIT %i\n",j);
			dst_unit->learn(src,scale);
		}
	}
}

// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Learner::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Learner> >();
    import_array(); // Important ! Sinon c'est seg fault
    class_<Learner>("Learner",
    "======================================================================\n"
    "\n"
    " A Learner manages the learning rules\n"  
    " The learning rules are defined by using the add(src,dst,parameters)  \n"
    " function \n"
    " Based on the equation  dwij/dt = F(wij,src[j],dst[i])\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize unit\n")
	)
	.def ("add", &Learner::add,
	      "add(src,dst,parameters) -- defines the learning rule between maps src and dst\n")
	.def("learn",&Learner::learn,
	     "learn(float scale = 1.0) -- Learns the weights with the defined learning rule\n")
        ;
}
