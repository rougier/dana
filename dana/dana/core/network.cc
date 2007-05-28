//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <boost/thread/thread.hpp>
#include "network.h"

using namespace dana::core;


// ============================================================================
//  constructor
// ============================================================================
Network::Network (void): Object (), width(1), height(1)
{}

// ============================================================================
//  destructor
// ============================================================================
Network::~Network (void)
{
    maps.clear();
}

// ============================================================================
//  append a new map
// ============================================================================
void
Network::append (MapPtr map)
{
    std::vector<MapPtr>::iterator result;
    result = find (maps.begin(), maps.end(), map);
    if (result != maps.end())
        return;
        
    maps.push_back (MapPtr (map));
    map->network = this;
    compute_geometry();
}


// ============================================================================
//  get map at index
// ============================================================================
MapPtr
Network::get (const int index)
{
    int i = index;

    if (i < 0)
        i += maps.size();
    try {
        return maps.at(i);
    } catch (...) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw_error_already_set();
    }
    return maps.at(0);
}


// ============================================================================
//  get size
// ============================================================================
int
Network::size (void) const
{
    return maps.size();
}

// ============================================================================
//  evaluate maps activity
// ============================================================================
void
Network::evaluate (unsigned long epochs, bool use_thread)
{
    if (use_thread) {
        boost::thread_group threads;
        barrier = new boost::barrier (maps.size());
        for (unsigned int i = 0; i < maps.size(); i++) {
            maps[i]->barrier = barrier;
            Map::map = maps[i].get();
            Map::epochs = epochs;
            threads.create_thread (&Map::evaluate);
        }
        threads.join_all();
        delete barrier;
     } else {
        for (unsigned long j=0; j<epochs; j++) {
           for (unsigned int i = 0; i < maps.size(); i++)
                maps[i]->compute_dp ();
           for (unsigned int i = 0; i < maps.size(); i++)
                maps[i]->compute_dw ();
        }
     }
}

// ============================================================================
//  Clear all maps
// ============================================================================
void
Network::clear (void)
{
    for (unsigned int i = 0; i < maps.size(); i++)
        maps[i]->clear();
}


// ============================================================================
//  get shape
// ============================================================================
object
Network::get_shape (void) 
{
    object shape = make_tuple (width, height);
    return shape;
}

// ============================================================================
//  compute network and map normalized geometry
// ============================================================================
void
Network::compute_geometry (void)
{
    if (!maps.size())
        return;

    // Get min/max position
    int xmin = maps[0]->x;
    int xmax = maps[0]->x;
    int ymin = maps[0]->y;
    int ymax = maps[0]->y;
    for (int i=0; i<size(); i++) {
        if (maps[i]->x < xmin) xmin = maps[i]->x;
        if (maps[i]->x > xmax) xmax = maps[i]->x;
        if (maps[i]->y < ymin) ymin = maps[i]->y;
        if (maps[i]->y > ymax) ymax = maps[i]->y;
    }
    
    // Array initialization (needed below)
    int column_size [xmax-xmin+1];
    int column_start [xmax-xmin+1];
    int line_size [ymax-ymin+1];
    int line_start [ymax-ymin+1];
    
    for (int i=0; i<(xmax-xmin+1); i++) {
        column_size[i] = 0;
        column_start[i] = 0;
    }
    for (int i=0; i<(ymax-ymin+1); i++) {
        line_size[i] = 0;
        line_start[i] = 0;
    }

    // Compute lines and columns size in terms of units
    //  (first run, without offset consideration)
    for (int i=0; i<size(); i++) {
        int x = maps[i]->x - xmin;
        int y = maps[i]->y - ymin;
        
        int w = maps[i]->dx + maps[i]->width * maps[i]->zoom;
        int h = maps[i]->dy + maps[i]->height * maps[i]->zoom;
        
        if (w > column_size[x])
            column_size[x] = w;
        if (h > line_size[y])
            line_size[y] = h;
    }
    
    
    // Compute line and column starts
    column_start[0] = 1;
    for (int i=1; i<(xmax-xmin+1); i++) {
        if (column_size[i-1])
            column_start[i] = column_start[i-1] + 1 + column_size[i-1];
        else
            column_start[i] = column_start[i-1];
    }
    
    line_start[0] = 1;
    for (int i=1; i<(ymax-ymin+1); i++) {
        if (line_size[i-1])
            line_start[i] = line_start[i-1] + 1 + line_size[i-1];
        else
            line_start[i] = line_start[i-1];
    }
    
    // Overall geometry
    float w = column_start[xmax-xmin] + column_size[xmax-xmin] + 1;
    float h = line_start[ymax-ymin] + line_size[ymax-ymin] + 1;
    width = (int) (w);
    height = (int) (h);
    
    // Set frame for all maps
    for (unsigned int i=0; i<maps.size(); i++) {
        int x = maps[i]->x - xmin;
        int y = maps[i]->y - ymin;
        
        object frame = make_tuple (
            (column_start[x] + maps[i]->dx) * 1.0f/w,
            (line_start[y]   + maps[i]->dy) * 1.0f/h,
            (maps[i]->width  * maps[i]->zoom )* 1.0f/w,                                   
            (maps[i]->height * maps[i]->zoom ) * 1.0f/h);
        maps[i]->set_frame (frame);
    }
}



// ============================================================================
//  python export
// ============================================================================

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(evaluate_overloads, evaluate, 0, 2)

void
Network::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Network> >();

    class_<Network> ("Network",
     "======================================================================\n"
    "\n"
    "A network is a set of maps that are evaluated synchronously.\n"
    "\n"
    "Attributes:\n"
    "-----------\n" 
    "   spec : Specification for the network\n"
    "   shape: Shape of the network (tuple, read_only) \n"
    "\n"
    "======================================================================\n",
        init<> (
        "__init__() -- initializes network\n")
        )
        
        .def ("__getitem__", &Network::get,
         "x.__getitem__ (y)  <==> x[y]\n\n"
        "Use of negative indices is supported.\n")
        
        .def ("__len__",     &Network::size,
        "__len__() -> integer -- return number of maps\n")
        
        .def ("append",      &Network::append,
        "append(map) -- append map to end\n")
        
        .def ("clear",       &Network::clear,
        "clear() -- clear map activity\n")

        .def ("evaluate",    &Network::evaluate,
        evaluate_overloads (args("n", "use_thread"), 
        "evaluate(n=1, use_thread=false) -- evaluate all maps for n epochs")
        )
         
        .def_readwrite ("spec", &Network::spec)
        .def_readonly ("shape", &Network::get_shape)
        ;
}
