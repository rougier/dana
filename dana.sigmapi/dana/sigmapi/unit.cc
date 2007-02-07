//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$

#include "core/map.h"
#include "core/layer.h"
#include "core/spec.h"
#include "link.h"
#include "unit.h"
#include <iostream>
#include <numpy/arrayobject.h>

using namespace boost::python::numeric;
using namespace dana::sigmapi;

// Constructor
// -----------------------------------------------------------------------------
Unit::Unit(void) : core::Unit()
{}

// Destructor
// -----------------------------------------------------------------------------
Unit::~Unit(void)
{}

// Connect
// -----------------------------------------------------------------------------
void
Unit::connect(core::LinkPtr link)
{
    afferents.push_back(link);
}



// Evaluate new potential and returns difference
// -----------------------------------------------------------------------------
float
Unit::compute_dp (void)
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
        input += afferents[i]->compute();
    }

    float lateral = 0;
    size = laterals.size();

    for (unsigned int i=0; i<size; i++)
    {
        lateral += laterals[i]->compute();
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

object
Unit::get_weights (const core::LayerPtr layer)
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
	const std::vector<core::LinkPtr> *wts;
	if (layer.get() == this->layer) {
		wts = &laterals;
	} else {
		wts = &afferents;
	}

	for (unsigned int i=0; i< wts->size(); i++) {
		core::UnitPtr unit = wts->at(i)->source;
		if(unit == 0)
		{
			// It means that wts->at(i) is a sigmapi::Link
			// in which the source neurons are managed in a different way
			unit = ((sigmapi::Link*)(wts->at(i).get()))->get_source(0);
		}
		if (unit->layer == layer.get())
			if ((unit->y > -1) && (unit->x > -1))
				data[unit->y*width+unit->x] += wts->at(i)->weight;
	}
	return extract<numeric::array>(obj);	
}

// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Unit::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();
    import_array();
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
                                    );
}
