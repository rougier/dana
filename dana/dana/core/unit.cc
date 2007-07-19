//
// Copyright (C) 2006-2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <cmath>
#include "unit.h"
#include "link.h"
#include "layer.h"
#include "map.h"


using namespace boost::python::numeric;
using namespace dana::core;

//_________________________________________________________________________Unit
Unit::Unit (float potential) : Object()
{
    this->potential = potential;
    x = -1;
    y = -1;
    layer = 0;
    laterals.clear();
    afferents.clear();
}

//________________________________________________________________________~Unit
Unit::~Unit(void)
{}

//___________________________________________________________________compute_dp
float
Unit::compute_dp (void)
{
    return 0.0f;
}

//___________________________________________________________________compute_dw
float
Unit::compute_dw (void)
{
    return 0.0f;
}

//______________________________________________________________________connect
void
Unit::connect (UnitPtr source, float weight, object data)
{
    LinkPtr link = LinkPtr (new Link (source, weight));

    if (source->layer == layer)
        laterals.push_back (link);
    else
        afferents.push_back (link);
}

//______________________________________________________________________connect
void
Unit::connect (UnitPtr source, float weight)
{
    connect (source, weight, object());
}

//______________________________________________________________________connect
void
Unit::connect (LinkPtr link)
{
    if (link->source->layer == layer)
        laterals.push_back (link);
    else
        afferents.push_back (link);
}

//________________________________________________________________________clear
void
Unit::clear (void)
{
    laterals.clear();
    afferents.clear();
}

//________________________________________________________________get/set layer
LayerPtr
Unit::get_layer (void)
{
    return LayerPtr(layer);
}

void
Unit::set_layer (Layer *layer)
{
    this->layer = layer;
}

//_____________________________________________________________get/set position
int
Unit::get_x (void)
{
    return x;
}

void
Unit::set_x (int x)
{
    this->x = x;
}

int
Unit::get_y (void)
{
    return y;
}

void
Unit::set_y (int y)
{
    this->y = y;
}

tuple
Unit::get_position (void)
{
    return make_tuple (x,y);
}

void
Unit::set_position (tuple position)
{
    try	{
        x = extract< int >(position[0])();
        y = extract< int >(position[1])();
    } catch (...) {
        PyErr_Print();
        return;
    }
}

void
Unit::set_position (int x, int y)
{
    this->x = x;
    this->y = y;
}

//_________________________________________________________________get/set spec
SpecPtr
Unit::get_spec (void)
{
    if ((spec == SpecPtr()) && (layer))
        return layer->get_spec();
    return SpecPtr(spec);
}

void
Unit::set_spec (SpecPtr sepc)
{
    this->spec = SpecPtr(spec);
}

//__________________________________________________________________get_weights
object
Unit::get_weights (LayerPtr layer)
{
    if (layer == object()) {
        PyErr_SetString(PyExc_AssertionError, "layer is None");
        throw_error_already_set();
        return object();
    }

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
                data[unit->y*width+unit->x] += wts->at(i)->weight;
    }
    return extract<numeric::array>(obj);  
}


//________________________________________________________________python_export
void
Unit::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();

    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");  

    // member function pointers for overloading
    void (Unit::*connect_src_data)(UnitPtr,float,object) = &Unit::connect;        
    void (Unit::*connect_src)(UnitPtr,float) = &Unit::connect;    
    void (Unit::*connect_link)(LinkPtr) = &Unit::connect;
    
    class_<Unit, bases <Object> >("Unit",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A unit is a potential that is computed on the basis of some external  \n"
    "sources that feed the unit. Those sources can be anything as long as  \n"
    "they have some potential.                                             \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "-----------                                                           \n"
    "   potential -- unit potential                                        \n"
    "   position  -- unit position within layer (read only)                \n"
    "   spec -- specification of parameters related to unit behavior       \n"
    "______________________________________________________________________\n",

    init < optional <float> > (
        (arg("potential") = 0.0f),
        "__init__ (potential=0)" ))
                
    .def_readwrite ("potential", &Unit::potential)
    .def_readonly  ("position", &Unit::get_position)
    .add_property  ("spec", &Unit::get_spec, &Unit::set_spec)
            
    .def ("compute_dp", &Unit::compute_dp,
          "compute_dp() -> float\n\n"
          "computes potential and return dp")

    .def ("compute_dw", &Unit::compute_dw,
          "compute_dw() -> float\n\n"
          "computes weights and returns dw")
    
    .def ("connect", connect_src)
    .def ("connect", connect_src_data)
    .def ("connect", connect_link,
          "connect (source, weight, data=None)\n\n"
          "connect to source using weight\n\n"  
          "connect (link)\n\n"
          "connect using link")

    .def ("clear", &Unit::clear,
          "clear ()\n\n"
          "remove all links")

    .def ("weights", &Unit::get_weights,
          "weights(layer)\n\n"
          "return weights from layer as a numpy::array")
        ;
}
