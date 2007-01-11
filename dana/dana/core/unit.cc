//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <cmath>
#include "unit.h"
#include "link.h"
#include "layer.h"
#include "map.h"


using namespace boost::python::numeric;
using namespace dana::core;


// =============================================================================
//  constructor
// =============================================================================
Unit::Unit(void) : Object()
{
    potential = 0.0;
    x = -1;
    y = -1;
    layer = 0;
    laterals.clear();
    afferents.clear();
}

// =============================================================================
//  destructor
// =============================================================================
Unit::~Unit(void)
{}

// =============================================================================
//  evaluates new potential and returns difference
// =============================================================================
float
Unit::evaluate (void)
{
    return 0.0f;
}

// =============================================================================
//  connect to src using weight w
// =============================================================================
void
Unit::connect (UnitPtr src, float w)
{
    LinkPtr link = LinkPtr (new Link (src, w));

    if (src->layer == layer)
        laterals.push_back (link);
    else
        afferents.push_back (link);
}

// =============================================================================
//  remove all links
// =============================================================================
void
Unit::clear (void)
{
    laterals.clear();
    afferents.clear();
}

// =============================================================================
//  get owning layer
// =============================================================================
LayerPtr
Unit::get_layer (void) const
{
    return LayerPtr(layer);
}

// =============================================================================
//  set owning layer
// =============================================================================
void
Unit::set_layer (Layer *l)
{
    layer = l;
}

// =============================================================================
//  get x position (relative to layer)
// =============================================================================
int
Unit::get_x (void) const
{
    return x;
}

// =============================================================================
//  set x position (relative to layer)
// =============================================================================
void
Unit::set_x (const int value)
{
    x = value;
}

// =============================================================================
//  get y position (relative to layer)
// =============================================================================
int
Unit::get_y (void) const
{
    return y;
}

// =============================================================================
//  set y position (relative to layer)
// =============================================================================
void
Unit::set_y (const int value)
{
    y = value;
}

// =============================================================================
//  get unit specification
// =============================================================================
object
Unit::get_spec (void) const
{
    if ( (!spec.ptr()) && (layer) )
        return layer->get_spec();
    return spec;
}

// =============================================================================
//  Set unit specification.
//  If given specification is none, specification from owning layer is used.
// =============================================================================
void
Unit::set_spec (const object s)
{
    if ( (!s.ptr()) && (layer) ) {
        spec = layer->get_spec();
    } else {
        spec = s;
    }
}

// =============================================================================
//  Get position as a tuple of int
// =============================================================================
object
Unit::get_position (void) const
{
    object position = make_tuple (x,y);
    return position;
}

// =============================================================================
//  Set position from a tuple of int
// =============================================================================
void
Unit::set_position (const object position)
{
    try	{
        x = extract< int >(position[0])();
        y = extract< int >(position[1])();
    } catch (...) {
        PyErr_Print();
        return;
    }
}

// =============================================================================
//  Set position
// =============================================================================
void
Unit::set_position (const int xx, const int yy)
{
    x = xx;
    y = yy;
}

// =============================================================================
//  Get weights from layer as a numpy::array
// =============================================================================
object
Unit::get_weights (const LayerPtr layer)
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
    const std::vector<LinkPtr> *wts;
    if (layer.get() == this->layer) {
        wts = &laterals;
    } else {
        wts = &afferents;
    }
    for (unsigned int i=0; i< wts->size(); i++) {
        UnitPtr unit = wts->at(i)->source;
        if (unit->layer == layer.get())
            if ((unit->y > -1) && (unit->x > -1))
                data[unit->y*width+unit->x] = wts->at(i)->weight;
    }
    return extract<numeric::array>(obj);  
}



// =============================================================================
//   boost wrapping code
// =============================================================================
void
Unit::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();

    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");  
    
    class_<Unit>("Unit",
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
    "   spec      : specification of the unit\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__ () -- initialize unit\n")
        )
                
        .def_readwrite ("potential", &Unit::potential)

        .def_readonly ("position", &Unit::get_position)
        
        .def ("evaluate", &Unit::evaluate,
        "evaluate() -> float -- evaluate new potential and return difference\n")
        
        .def ("connect", &Unit::connect,
        "connect (src, w) -- connect to src using weight w\n")

        .def ("clear", &Unit::clear,
        "clear () -- remove all links\n")

        .def ("weights", &Unit::get_weights,
        "weights(layer) -- return weights from layer as a numpy::array\n")
        ;
}
