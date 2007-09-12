//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id: unit.cc 262 2007-08-06 12:02:43Z fix $

#include "dana/core/map.h"
#include "dana/core/layer.h"
#include "dana/core/spec.h"
#include "dana/cnft/spec.h"
#include "link.h"
#include "unit.h"
#include <iostream>
#include <numpy/arrayobject.h>

using namespace boost::python::numeric;
using namespace dana::sigmapi::core;

// Constructor
// -----------------------------------------------------------------------------
Unit::Unit(void) : dana::core::Unit()
{
    input = 0;
}

// Destructor
// -----------------------------------------------------------------------------
Unit::~Unit(void)
{}

// Connect
// -----------------------------------------------------------------------------
void
Unit::connect(dana::core::LinkPtr link)
{
    afferents.push_back(link);
}



// Evaluate new potential and returns difference
// -----------------------------------------------------------------------------
float
Unit::compute_dp (void)
{
    dana::core::SpecPtr sp = layer->get_spec();
    if (sp.get() == NULL)
        return 0.0f;
    dana::cnft::Spec *s = dynamic_cast<cnft::Spec *>(sp.get());
    if (s == 0)
        return 0.0f;

    float tau      = s->tau;
    float alpha    = s->alpha;
    float baseline = s->baseline;
    float min_act  = s->min_act;
    float max_act  = s->max_act;    

    float input = 0;
    unsigned int size = afferents.size();

    for (unsigned int i=0; i<size; i++)
    {
        input += afferents[i]->compute();
    }

    float lateral = 0;
    size = laterals.size();

    for (unsigned int i=0; i<size; i++)
    {
        lateral += laterals[i]->compute();
    }
    this->input = input ; //(1.0f/alpha)*(lateral + input);
    float du = (-potential + baseline + (1.0f/alpha)*(lateral + input)) / tau;
    float value = potential;
    potential += du;

    if (potential < min_act)
        potential = min_act;

    if (potential > max_act)
        potential = max_act;

    return value-potential;
}

object
Unit::get_weights (const dana::core::LayerPtr layer)
{
	if ((layer->map == 0) || (layer->map->width == 0)) {
		PyErr_SetString(PyExc_AssertionError, "layer has no shape");
		throw_error_already_set();
		return object();
	}

	unsigned int width = layer->map->width;
	unsigned int height = layer->map->height;
	
	npy_intp dims[2] = {height, width};
	object obj(handle<>(PyArray_SimpleNew (2, dims, PyArray_FLOAT)));

	PyArrayObject *array = (PyArrayObject *) obj.ptr();

	PyArray_FILLWBYTE(array, 0);

	float *data = (float *) array->data;
	const std::vector<dana::core::LinkPtr> *wts;
	if (layer.get() == this->layer) {
		wts = &laterals;
	} else {
		wts = &afferents;
	}

	for (unsigned int i=0; i< wts->size(); i++) {
		dana::core::UnitPtr unit = wts->at(i)->source;
		if(unit == 0)
		{
			// It means that wts->at(i) is a sigmapi::Link
			// in which the source neurons are managed in a different way
			unit = ((dana::sigmapi::core::Link*)(wts->at(i).get()))->get_source(0);
		}
		if (unit->layer == layer.get())
			if ((unit->y > -1) && (unit->x > -1))
				data[unit->y*width+unit->x] += wts->at(i)->weight;
	}
	return extract<boost::python::numeric::array>(obj);	
}

/*int
Unit::count_connections(void)
{
    int numb = 0;
    for (unsigned int i=0; i<afferents.size(); i++)
    {
        numb += afferents[i]->count_connections();
    }
    for (unsigned int i=0; i<laterals.size(); i++)
    {
        numb += laterals[i]->count_connections();
    }
    return numb;
}*/


// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Unit::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();
    import_array();
    class_<Unit, bases<dana::core::Unit> >("Unit",
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
                                         "__init__ () -- initialize unit\n"))
				    .def_readonly ("input", &Unit::get_input)
                                    ;
}
