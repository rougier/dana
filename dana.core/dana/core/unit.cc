//
// Copyright (C) 2006-2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: unit.cc 257 2007-07-29 11:38:44Z rougier $
// _____________________________________________________________________________

#include <iostream>
#include "unit.h"
#include "link.h"
#include "layer.h"
#include "map.h"
#include <numpy/arrayobject.h>
#include <numpy/arrayobject.h>

using namespace dana::core;

// _________________________________________________________________________Unit
Unit::Unit (float potential) : Object()
{
    this->potential = potential;
    x = -1;
    y = -1;
    layer = 0;
    laterals.clear();
    afferents.clear();
}

// ________________________________________________________________________~Unit
Unit::~Unit(void)
{
    laterals.clear();
    afferents.clear();
}

// ________________________________________________________________________write
int
Unit::write (xmlTextWriterPtr writer)
{
    xmlTextWriterWriteFormatAttribute (writer, BAD_CAST "potential", "%f", potential);

    for (unsigned int i=0; i< laterals.size(); i++) {
        write_element (writer, "Link", laterals[i]);

        laterals[i]->write(writer);
        
        xmlTextWriterEndElement (writer);
    }

    for (unsigned int i=0; i< afferents.size(); i++) {
        write_element (writer, "Link", afferents[i]);

        afferents[i]->write(writer);

        xmlTextWriterEndElement (writer);
    }

    return 0;
}

// _________________________________________________________________________read
int
Unit::read (xmlTextReaderPtr reader)
{
    
    printf("id: %s\n", read_attribute (reader, "id").c_str());
    printf("potential: %s\n", read_attribute (reader, "potential").c_str());


    int ret = 1;
    while (ret == 1) {
        std::string name = (char *) xmlTextReaderConstName(reader);

        if ((xmlTextReaderNodeType(reader) == XML_ELEMENT_DECL) && (name == "Unit"))
            return 0;
        
        if ((xmlTextReaderNodeType(reader) == XML_ELEMENT_NODE) && (name == "Link")) {
            std::string klassname  = read_attribute (reader, "class");
            std::string modulename = read_attribute (reader, "module");

            // Retrieve the main module
            py::object main = py::import("__main__");
  
            // Retrieve the main module's namespace
            py::object global (main.attr("__dict__"));

            // Build python code to create the requested Object (some Link)
            std::string code = std::string ("import ") + modulename + std::string ("\n");
            code += std::string("__dana_generated_object = ")
                + modulename + std::string(".") + klassname + std::string ("()\n");

            // Execute code
            py::exec (code.c_str(), global, global);            

            LinkPtr link = py::extract<LinkPtr>(global["__dana_generated_object"]);
        }
        ret = xmlTextReaderRead(reader);
    }
    
    xmlTextReaderNext(reader);
    return 0;
}

// ________________________________________________________________get_afferents
py::list
Unit:: get_afferents (void)
{
    py::list l;
    for (unsigned int i=0; i<afferents.size(); i++)
        l.append (afferents[i]->get());
    return l;
}

// ________________________________________________________________get_laterals
py::list
Unit::get_laterals (void)
{
    py::list l;
    for (unsigned int i=0; i<laterals.size(); i++)
        l.append (laterals[i]->get());
    return l;
}

//_________________________________________________________________get_potential
float
Unit:: get_potential (void)
{
    return potential;
}

//_________________________________________________________________set_potential
void
Unit::set_potential (float potential)
{
    this->potential = potential;
}

//_____________________________________________________________________get_layer
LayerPtr
Unit::get_layer (void)
{
    return LayerPtr (layer);
}

//_____________________________________________________________________set_layer
void
Unit::set_layer (Layer *layer)
{
    this->layer = layer;
}

//______________________________________________________________________get_spec
SpecPtr
Unit::get_spec (void)
{
    return SpecPtr(spec);
}

//______________________________________________________________________set_spec
void
Unit::set_spec (SpecPtr spec)
{
    this->spec = SpecPtr(spec);
}

//__________________________________________________________________get_position
py::tuple
Unit::get_position (void)
{
    return py::make_tuple (x,y);
}

//__________________________________________________________________set_position
void
Unit::set_position (py::tuple position)
{
    try {
        x = py::extract <int> (position[0])();
        y = py::extract <int> (position[1])();
    } catch (...) {
        PyErr_Print();
        return;
    }
}

//__________________________________________________________________set_position
void
Unit::set_position (int x, int y)
{
    this->x = x;
    this->y = y;
}

//_________________________________________________________________________get_x
int 
Unit::get_x (void)
{
    return x;
}

//_________________________________________________________________________set_x
void
Unit::set_x (int x)
{
    this->x = x;
}

//_________________________________________________________________________get_y
int 
Unit::get_y (void)
{
    return y;
}

//_________________________________________________________________________set_y
void
Unit::set_y (int y)
{
    this->y = y;
}

//_____________________________________________________________________operator=
Unit &
Unit::operator= (const Unit &other)
{
    if (this == &other)
        return *this;
    this->potential = other.potential;
    return *this;
}

//_____________________________________________________________________operator+
Unit const 
Unit::operator+ (Unit const &other) const
{
    return Unit (potential+other.potential);
}

//_____________________________________________________________________operator-
Unit const
Unit::operator- (Unit const &other) const
{
    return Unit(potential-other.potential);
}

//_____________________________________________________________________operator*
Unit const
Unit::operator* (Unit const &other) const
{
    return Unit (potential*other.potential);
}

//_____________________________________________________________________operator/
Unit const
Unit::operator/ (Unit const &other) const
{
    return Unit (potential/other.potential);
}

//_____________________________________________________________________operator+
Unit const
Unit::operator+ (float value) const
{
    return Unit (potential + value);
}


//_____________________________________________________________________operator-
Unit const
Unit::operator- (float value) const
{
    return Unit (potential - value);
}


//_____________________________________________________________________operator*
Unit const
Unit::operator* (float value) const
{
    return Unit (potential * value);
}


//_____________________________________________________________________operator/
Unit const
Unit::operator/ (float value) const
{
    return Unit (potential / value);
}

//____________________________________________________________________operator+=
Unit &
Unit::operator+= (Unit const &other)
{
    potential += other.potential;
    return *this;
}

//____________________________________________________________________operator-=
Unit &
Unit::operator-= (Unit const &other)
{
    potential -= other.potential;
    return *this;
}

//____________________________________________________________________operator*=
Unit &
Unit::operator*= (Unit const &other)
{
    potential *= other.potential;
    return *this;
}

//____________________________________________________________________operator/=
Unit &
Unit::operator/= (Unit const &other)
{
    potential /= other.potential;
    return *this;
}

//____________________________________________________________________operator+=
Unit &
Unit::operator+= (float value)
{
    potential += value;
    return *this;
}

//____________________________________________________________________operator-=
Unit &
Unit::operator-= (float value)
{
    potential -= value;
    return *this;
}

//____________________________________________________________________operator*=
Unit &
Unit::operator*= (float value)
{
    potential *= value;
    return *this;
}

//____________________________________________________________________operator/=
Unit &
Unit::operator/= (float value)
{
    potential /= value;
    return *this;
}

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
Unit::connect (UnitPtr source, float weight, py::object data)
{
    LinkPtr link = LinkPtr (new Link (source, weight));

    if ((source.get() == this) || ((layer) && (source->layer == layer)))
        laterals.push_back (link);
    else
        afferents.push_back (link);
}

//______________________________________________________________________connect
void
Unit::connect (UnitPtr source, float weight)
{
    connect (source, weight, py::object());
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


//__________________________________________________________________get_weights
py::object
Unit::get_weights (LayerPtr layer)
{
    if (layer == py::object()) {
        PyErr_SetString(PyExc_AssertionError, "layer is None");
        py::throw_error_already_set();
        return py::object();
    }

    if ((layer->map == 0) || (layer->map->width == 0)) {
        PyErr_SetString(PyExc_AssertionError, "layer has no shape");
        py::throw_error_already_set();
        return py::object();
    }

    unsigned int width = layer->map->width;
    unsigned int height = layer->map->height;

    npy_intp dims[2] = {height, width};
    py::object obj(py::handle<>(PyArray_SimpleNew (2, dims, PyArray_FLOAT)));
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
    return py::extract<numeric::array>(obj);  
}


// _________________________________________________________some more arithmetic
Unit const operator+ (float value, Unit const &other)
{ return Unit(value+other.potential); }
Unit const operator- (float value, Unit const &other)
{ return Unit(value-other.potential); }
Unit const operator* (float value, Unit const &other)
{ return Unit(value*other.potential); }
Unit const operator/ (float value, Unit const &other)
{ return Unit(value/other.potential); }


// __________________________________________________________________UnitWrapper
//
// This ensures that some methods are overridable from within python
//
class UnitWrapper:  public Unit, public py::wrapper<Unit> {
public:
    
    UnitWrapper (float potential = 0.0f) : Unit (potential) {};

    // _______________________________________________________________compute_dp
    float compute_dp (void)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        if (py::override compute_dp = this->get_override("compute_dp")) {
            float result = compute_dp();
            PyGILState_Release(gstate);
            return result;
        }
        float result = Unit::compute_dp();
        PyGILState_Release(gstate);
        return result;        
    }
    float default_compute_dp() { return this->Unit::compute_dp(); }

    // _______________________________________________________________compute_dw
    float compute_dw (void)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        if (py::override compute_dw = this->get_override("compute_dw")) {
            float result = compute_dw();
            PyGILState_Release(gstate);
            return result;
        }
        float result = Unit::compute_dw();
        PyGILState_Release(gstate);
        return result;        
    }
    float default_compute_dw() { return this->Unit::compute_dw(); }
};


//________________________________________________________________python_export
void
Unit::python_export (void) {
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Unit> >();

    import_array();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    // member function pointers for overloading
    void (Unit::*connect_src_data)(UnitPtr,float,object) = &Unit::connect;        
    void (Unit::*connect_src)(UnitPtr,float) = &Unit::connect;    
    void (Unit::*connect_link)(LinkPtr) = &Unit::connect;
    
    class_<UnitWrapper, bases <Object>, boost::noncopyable>("Unit",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A unit is a potential that is computed on the basis of some external  \n"
    "sources that feed the unit. Those sources can be anything as long as  \n"
    "they possess some potential. The potential is evaluated with a call to\n"
    "'update_dp()' while links are updated with a call to 'update_dw()'.   \n"
    "Note that these two methods can be overriden from within python. Unit \n"
    "also implements the float interface and may be manipulated like any   \n"
    "float.                                                                \n"
    "______________________________________________________________________\n",

    init < optional <float> > (
        (arg("potential") = 0.0f),
        "__init__ (potential=0)" ))
                
        .add_property ("potential",
                       &Unit::get_potential, &Unit::set_potential,
                       "membrane potential")

        .add_property ("position",
                       &Unit::get_position,
                       "position within layer (read only)")

        .add_property ("spec",
                       &Unit::get_spec, &Unit::set_spec,
                       "specifications that governs unit potential evaluation")

        .add_property ("laterals",
                       &Unit::get_laterals,
                       "list of lateral links (read only)")

        .add_property ("afferents",
                       &Unit::get_afferents,
                       "list of afferent links (read only)")

        .def ("weights", &Unit::get_weights,
              "weights(layer)\n"
              "return weights from layer as a numpy::array")

        .def ("__float__", &Unit::get_potential)

        .def ("compute_dp", &Unit::compute_dp, &UnitWrapper::default_compute_dp,
              "compute_dp() -> float\n"
              "computes potential and return dp")

        .def ("compute_dw", &Unit::compute_dw, &UnitWrapper::default_compute_dw,
              "compute_dw() -> float\n"
              "computes weights and returns dw")
    
        .def ("connect", connect_src)
        .def ("connect", connect_src_data)
        .def ("connect", connect_link,
              "connect (source, weight, data=None)\n"
              "connect (link)\n"
              "connect to source using weight or connect using link")
        
        .def ("clear", &Unit::clear,
              "clear ()\n"
              "remove all links")

        
        .def(self + self)
        .def(self + float())
        .def(float() + self)
        .def(self += self)
        .def(self += float())

        .def(self - self)
        .def(self - float())
        .def(float() - self)
        .def(self -= self)
        .def(self -= float())

        .def(self * self)
        .def(self * float())
        .def(float() * self)
        .def(self *= self)
        .def(self *= float())

        .def(self / self)
        .def(self / float())
        .def(float() / self)
        .def(self /= self)
        .def(self /= float())            
        ;
}
