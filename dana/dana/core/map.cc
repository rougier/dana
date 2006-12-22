//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#include <algorithm>
#include "network.h"
#include "map.h"
#include "layer.h"
#include "unit.h"

using namespace dana::core;


// =============================================================================
//  
// =============================================================================
unsigned long Map::epochs = 0;

// =============================================================================
//  destructor
// =============================================================================
Map *Map::map = 0;

// =============================================================================
//  constructor
// =============================================================================
Map::Map (object shape, object position) : Object()
{
    layers.clear();
    network = 0;
    set_position (position);
    set_shape (shape);
    shuffle_index = 0;
}

// =============================================================================
//  destructor
// =============================================================================
Map::~Map (void)
{}

// =============================================================================
//  append a new layer
// =============================================================================
void
Map::append (LayerPtr layer)
{
    std::vector<LayerPtr>::iterator result;
    result = find (layers.begin(), layers.end(), layer);
    if (result != layers.end())
        return;
        
    layers.push_back (LayerPtr (layer));
    layer->set_map (this);
    for (int i=0; i < layer->size(); i++) {
        layer->units[i]->set_position (i % width, i / width);
    }
}

// =============================================================================
//  get layer at index
// =============================================================================
LayerPtr
Map::get (const int index) const
{
    int i = index;

    if (i < 0)
        i += layers.size();
    try {
        return LayerPtr(layers.at(i));
    } catch (...) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw_error_already_set();
    }
}

// =============================================================================
//  get size
// =============================================================================
int
Map::size (void) const
{
    return layers.size();
}

// =============================================================================
//  remove all layer
// =============================================================================
void
Map::clear (void)
{
    width = 0;
    height = 0;
    layers.clear();
}

// =============================================================================
//  get unit at index from layer 0
// =============================================================================
UnitPtr
Map::unit (const int index) const
{
    return layers[0]->get (index);
}

// =============================================================================
//  get unit at x,y from layer 0
// =============================================================================
UnitPtr
Map::unit (const int x, const int y) const
{
    return layers[0]->get (x,y);
}

// =============================================================================
//  fill layer 0
// =============================================================================
int
Map::fill (object type)
{
    return layers[0]->fill(type);
}
            
// =============================================================================
//  get layer 0 potentials as a nupy::array
// =============================================================================
object
Map::get_potentials (void) const
{
    return layers[0]->get_potentials();
}

// =============================================================================
//  layers evaluation
// =============================================================================
void
Map::evaluate (void)
{
    shuffle_index = (shuffle_index + 1) % (shuffles.size());

    for (int i=0; i< size(); i++)
        layers[i]->evaluate ();
}

// =============================================================================
//   Static layers evaluation
// =============================================================================
void
Map::static_evaluate (void)
{
    if (map && map->network) {
        Map *m = map;
        Network *net = m->network;
        for (unsigned long i=0; i<epochs; i++) {
            map->evaluate ();
            net->barrier->wait();
        }
    }
}

// =============================================================================
//  get map specification
// =============================================================================
object
Map::get_spec (void) const
{
    return spec;
}

// =============================================================================
//  Set map specification.
//  If given specification is none, specification from owning network is used.
// =============================================================================
void
Map::set_spec (const object s)
{
    if ( (!s.ptr()) && (network) ) {
//        spec = network->get_spec();
    } else {
        spec = s;
    }
}

// =============================================================================
//  get shape as a tuple of int
// =============================================================================
object
Map::get_shape (void) const
{
    object shape = make_tuple (width, height);
    return shape;
}

// =============================================================================
//  set shape from a tuple of int
// =============================================================================
void
Map::set_shape (const object shape)
{
    int w, h;
    try	{
        w = extract< int >(shape[0])();
        h = extract< int >(shape[1])();
    } catch (...) {
        PyErr_Print();
        return;
    }

    set_shape (w,h);
}

// =============================================================================
//  set shape
// =============================================================================
void
Map::set_shape (const int w, const int h)
{
    width = w;
    height = h;
    for (unsigned int i = 0; i<layers.size(); i++)
        layers[i]->clear();

    // Generates shuffles for random evaluation
    std::vector<int> s;
    for (int i=0; i<w*h; i++)
        s.push_back (i);
    for (int i=0; i < 20; i++) {
        random_shuffle (s.begin(), s.end());
        shuffles.push_back(s);
    }

    if (network)
        network->compute_geometry();
}

// =============================================================================
//  get position as a tuple of int
// =============================================================================
object
Map::get_position (void) const
{
    object position = make_tuple (x, y);
    return position;
}

// =============================================================================
//  
// =============================================================================
void
Map::set_position (const object position)
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
//  
// =============================================================================
void
Map::set_position (const int xx, const int yy)
{
    x = xx;
    y = yy;
}

// =============================================================================
//  
// =============================================================================
object
Map::get_frame (void) const
{
    return frame;
}

// =============================================================================
//  
// =============================================================================
void
Map::set_frame (const object f)
{
    frame = f;
}



// =============================================================================
//    Boost wrapping code
// =============================================================================
void
Map::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Map> >();
    
    // member function pointers for overloading
    UnitPtr    (Map::*unit_index)(int) const = &Map::unit;
    UnitPtr    (Map::*unit_xy)(int, int) const = &Map::unit;
    void       (Map::*set_shape_object)(const object) = &Map::set_shape;
    void       (Map::*set_position_object)(const object) = &Map::set_position;
    
    class_<Map>("Map",
    "======================================================================\n"
    "\n"
    "A map is a set of layers that are evaluated synchronously.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "   spec :    Specification for the map\n"
    "   shape:    Shape of the map (tuple) \n"
    "   position: Position of the map (tuple) \n"
    "   frame:    Normalized frame containing the map (tuple, read-only)\n"
    "\n"
    "======================================================================\n",
    
        init<optional <tuple,tuple> > (
        "__init__(shape, position) -- initializes map\n")
        )
 
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
        
        
         .def_readwrite ("spec", &Map::spec)
               
        .add_property ("shape",  &Map::get_shape, set_shape_object)
        
        .add_property ("position",  &Map::get_position, set_position_object)
        
        .def_readonly ("frame", &Map::get_frame)
        
        ;
}

