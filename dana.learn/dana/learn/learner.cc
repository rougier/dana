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
#include <boost/python/detail/api_placeholder.hpp>

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
//  Defines the source layer of the links to learn
// =============================================================================
void
Learner::set_source(core::LayerPtr src)
{
	this->src = src;
}

// =============================================================================
//  Defines the destination layer of the links to learn 
// =============================================================================
void
Learner::set_destination(core::LayerPtr dst)
{
	this->dst = dst;
}

// =============================================================================
//  Add one coefficient for a tuple vi^powi * vj^powj
// =============================================================================

void
Learner::add_one(boost::python::list params)
{
	if(!PyList_Check(params.ptr())){
		PyErr_SetString(PyExc_ValueError, "expected a PyArrayObject");
		throw_error_already_set();
	}
 	boost::python::list poly_w = extract<boost::python::list>(params[2]);
	float value;
	int l_poly_w = boost::python::len(poly_w);
	std::vector<float> params_w;
	params_w.push_back((float)(extract<int>(params[0])));
	params_w.push_back((float)(extract<int>(params[1])));
	for(int i = 0 ; i < l_poly_w ; i++)
	{
		value = extract<float>(poly_w[i]);
		params_w.push_back(value);
	}
	learn_params.push_back(params_w);
}

// =============================================================================
//  Add the learning rules & the maps src and dst to the rules vector
// =============================================================================

void 
Learner::connect(void)
{
	learnStr learn;
	learn.source = src;
	learn.destination = dst;
 	learn.params = learn_params;
	learns.push_back(learn);
	learn_params.clear();
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
    " wij is the weight between presynaptic neuron j and postsynaptic neuron j\n"
    " To define a learning rule :\n"
    "             1) Set the source layer with set_source\n"
    "             2) Set the destination layer with set_destination\n"
    "             3) Add the elementary blocks of the learning with add_one([powi,powj,[polynomial function of w]])\n"
    " An elementary block of the learning rule is defined with powi, powj and an array representing a polynomial\n"
    " function of the weight : P(w).(vi**powi).(vj**powj)\n"
    " The array defining P(w) is constructed as following : [polynomial function of w] = [a0,a1,a2,...]\n"
    " with : P(w) = sum_i (ai.w**i)\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize unit\n")
	)
	.def ("set_source",&Learner::set_source,
	      "set_source(layer src) -- defines the source layer of the weights to learn\n")
	.def ("set_destination",&Learner::set_destination,
	      "set_destination(layer dst) -- defines the destination layer of the weights to learn\n")
	.def ("add_one",&Learner::add_one,
	      "add_one([powi,powj,[params]]) -- add one element of the learning rule\n")
	.def ("connect",&Learner::connect,
	      "connect() -- add the defined learning rule to the set of rules\n")
	.def("learn",&Learner::learn,
	     "learn(float scale = 1.0) -- Learns the weights with the defined learning rule\n")
        ;
}
