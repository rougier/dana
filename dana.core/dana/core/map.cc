//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: map.cc 249 2007-07-19 15:30:16Z rougier $

#include <algorithm>
#include "network.h"
#include "map.h"
#include "layer.h"
#include "unit.h"

using namespace dana::core;


// ============================================================================
//  constructor
// ============================================================================
Map::Map (py::object shape, py::object position) : Object()
{
    layers.clear();
    network = 0;
    spec.reset();
    set_position (position);
    set_shape (shape);
    frame = py::make_tuple (0,0,1,1);
    shuffle_index = 0;
}

// ============================================================================
//  destructor
// ============================================================================
Map::~Map (void)
{}

// ============================================================================
//  append a new layer
// ============================================================================
void
Map::append (LayerPtr layer)
{
    std::vector<LayerPtr>::iterator result;
    result = find (layers.begin(), layers.end(), layer);
    if (result != layers.end())
        return;
    
    layers.push_back (LayerPtr (layer));
    layer->set_map (this);
    for (int i=0; i < layer->size(); i++)
        layer->units[i]->set_position (i % width, i / width);
}

// ============================================================================
//  get layer at index
// ============================================================================
LayerPtr
Map::get (int index) 
{
    int i = index;

    if (i < 0)
        i += layers.size();
    try {
        return LayerPtr(layers.at(i));
    } catch (...) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        py::throw_error_already_set();
    }
    return LayerPtr();
}

// ============================================================================
//  get size
// ============================================================================
int
Map::size (void) 
{
    return layers.size();
}

// ============================================================================
//  get unit at index from layer 0
// ============================================================================
UnitPtr
Map::unit (int index) 
{
    return layers[0]->get (index);
}

// ============================================================================
//  get unit at x,y from layer 0
// ============================================================================
UnitPtr
Map::unit (int x,  int y) 
{
    return layers[0]->get (x,y);
}

// ============================================================================
//  fill layer 0
// ============================================================================
int
Map::fill (py::object type)
{
    return layers[0]->fill(type);
}
            
// ============================================================================
//  get layer 0 potentials as a nupy::array
// ============================================================================
py::object
Map::get_potentials (void) 
{
    return layers[0]->get_potentials();
}

// ============================================================================
//  Clear layer activities
// ============================================================================
void
Map::clear (void)
{
    for (unsigned int i=0; i < layers.size(); i++)
        layers[i]->clear();
}

// ============================================================================
//  compute potentials
// ============================================================================
void
Map::compute_dp (void)
{
    for (int i=0; i< size(); i++) {
        shuffle_index = int ( (rand()/float(RAND_MAX)) * shuffles.size() );
        layers[i]->compute_dp ();
    }
}

// ============================================================================
//  compute weights
// ============================================================================
void
Map::compute_dw (void)
{
    for (int i=0; i< size(); i++) {
        shuffle_index = int ( (rand()/float(RAND_MAX)) * shuffles.size() );
        layers[i]->compute_dw ();
    }
}

// ________________________________________________________________________write
int
Map::write (xmlTextWriterPtr writer)
{
    // <Map>
    xmlTextWriterStartElement (writer, BAD_CAST "Map");
    
    for (unsigned int i=0; i< layers.size(); i++)
        layers[i]->write(writer);

    // </Map>
    xmlTextWriterEndElement (writer);

    return 0;
}

// _________________________________________________________________________read
int
Map::read (xmlTextReaderPtr reader)
{
    xmlReaderTypes type   = XML_READER_TYPE_NONE;
    std::string    name   = "";
    int            status = 1;

    unsigned int index = 0;
    do  {
        status = xmlTextReaderRead(reader);
        if (status != 1)
            break;
        name = (char *) xmlTextReaderConstName(reader);
        type = (xmlReaderTypes) xmlTextReaderNodeType(reader);

        if ((type == XML_READER_TYPE_END_ELEMENT) && (name == "Map"))
            break;

        if ((type == XML_READER_TYPE_ELEMENT) && (name == "Layer")) {
            if (index < layers.size())
                layers[index]->read (reader);
            index++;
        }        
    } while (status == 1);    
    return 0;
}

// __________________________________________________________________get_network
Network *
Map::get_network (void)
{
    return network;
}
// __________________________________________________________________set_network
void
Map::set_network (class Network *network)
{
    this->network = network;
}

// _____________________________________________________________________get_spec
SpecPtr
Map::get_spec (void) 
{
    if ((spec == SpecPtr()) && (network)) {
        return network->get_spec();
    }
    return SpecPtr(spec);
}

// _____________________________________________________________________set_spec
void
Map::set_spec (SpecPtr spec)
{
    this->spec = SpecPtr(spec);
}

// ============================================================================
//  get shape as a tuple of int
// ============================================================================
py::object
Map::get_shape (void) 
{
    py::object shape = py::make_tuple (width, height);
    return shape;
}

// ============================================================================
//  set shape from a tuple of int
// ============================================================================
void
Map::set_shape (py::object shape)
{
    int w, h;
    try	{
        w = py::extract< int >(shape[0])();
        h = py::extract< int >(shape[1])();
    } catch (...) {
        PyErr_Print();
        return;
    }
    set_shape (w,h);
}

// ============================================================================
//  set shape
// ============================================================================
void
Map::set_shape (int w,  int h)
{
    width = w;
    height = h;
    for (unsigned int i = 0; i<layers.size(); i++)
        layers[i]->clear();

    // Generates shuffles for random evaluation
    std::vector<int> s(w*h, 0);
    for (int i=0; i<w*h; i++)
        s[i] = i;
    for (int i=0; i < 100; i++) {
        random_shuffle (s.begin(), s.end());
        shuffles.push_back(s);
    }
    
    if (network)
        network->compute_geometry();
}

// ============================================================================
//  get position as a tuple of int
// ============================================================================
py::object
Map::get_position (void) 
{
    py::object position = py::make_tuple (x, y);
    return position;
}

// ============================================================================
//  
// ============================================================================
void
Map::set_position (py::object position)
{
    try	{
        int size = py::extract< int > (position.attr("__len__")());
        x = py::extract< int >(position[0])();
        y = py::extract< int >(position[1])();
        dx = 0;
        dy = 0;
        zoom = 1;
        if (size == 3) {
            zoom = py::extract< int >(position[2])();
        } else if (size == 4) {
            dx = py::extract< int >(position[2])();
            dy = py::extract< int >(position[3])();        
        } else if (size == 5) {
            dx = py::extract< int >(position[2])();
            dy = py::extract< int >(position[3])();        
            zoom = py::extract< int >(position[4])();
        }

    } catch (...) {
        PyErr_Print();
        return;
    }
}

// ============================================================================
//  
// ============================================================================
void
Map::set_position (int x,  int y)
{
    this->x = x;
    this->y = y;
}

// ============================================================================
//  
// ============================================================================
py::object
Map::get_frame (void) 
{
    return frame;
}

// ============================================================================
//  
// ============================================================================
void
Map::set_frame (py::object f)
{
    frame = f;
}



// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Map::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Map> >();
    
    // member function pointers for overloading
    UnitPtr    (Map::*unit_index)(int)  = &Map::unit;
    UnitPtr    (Map::*unit_xy)(int, int)  = &Map::unit;
    void       (Map::*set_shape_object)( object) = &Map::set_shape;
    void       (Map::*set_position_object)( object) = &Map::set_position;
    
    class_<Map, bases <Object> >("Map",
    "======================================================================\n"
    "\n"
    "A map is a set of layers that are evaluated synchronously.\n"
    "\n"
    "\n"
    "======================================================================\n",
    
        init<optional <tuple,tuple> > (
            "__init__(shape, position)\n")
        )

        // Properties
        .add_property ("spec",
                       &Map::get_spec, &Map::set_spec,
                       "Parameters of the map")

        
        .def ("compute_dp", &Map::compute_dp,
        "compute_dp() -> float -- computes potentials and return dp\n")

        .def ("compute_dw", &Map::compute_dw,
        "compute_dw() -> float -- computes weights and returns dw\n")

        .def ("clear", &Map::clear,
        " clear() -- clear layers activity\n")

        .def ("__getitem__", &Map::get,
        "x.__getitem__ (y)  <==> x[y]\n\n"
        "Use of negative indices is supported.\n")

        .def ("layer", &Map::get,
        "x.__getitem__ (y)  <==> x[y]\n\n"
        "Use of negative indices is supported.\n")

        .def ("__len__", &Map::size,
        "__len__() -> integer -- return number of layers\n")

        .def ("append", &Map::append,
        "append(layer) -- append layer to end\n")

        .def ("unit", unit_index,
        "unit(index) -> unit -- get unit of layer 0 at index\n\n"
        "Use of negative indices is supported.\n")

        .def ("unit", unit_xy,
        "unit(x,y) -> unit -- get unit of layer 0 at (x,y) coordinates\n\n"
        "Use of negative indices is supported.\n")

        .def ("fill", &Map::fill,
        "fill(type) -> integer -- fill layer 0 with type object")

        .def ("potentials", &Map::get_potentials,
        "potentials() -> numpy::array -- "
        "get units potential from layer 0 as an array")
        
        
        .add_property  ("shape",  &Map::get_shape, set_shape_object)
        .add_property  ("position",  &Map::get_position, set_position_object)
        .add_property  ("frame", &Map::get_frame, &Map::set_frame)
        
        ;
}

