//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <boost/thread/thread.hpp>
#include "projector.h"


using namespace dana::projection;

// =============================================================================
//  constructor
// =============================================================================
Projector::Projector (void) : object()
{}

// =============================================================================
//  destructor
// =============================================================================
Projector::~Projector (void)
{}

// =============================================================================
//  append a new layer
// =============================================================================
void
Projector::append (ProjectionPtr proj)
{
    std::vector<ProjectionPtr>::iterator result;
    result = find (projections.begin(), projections.end(), proj);
    if (result != projections.end())
        return;
        
    projections.push_back (ProjectionPtr (proj));
}

// =============================================================================
//  get projection at index
// =============================================================================
ProjectionPtr
Projector::get (const int index) const
{
    int i = index;

    if (i < 0)
        i += projections.size();
    try {
        return ProjectionPtr(projections.at(i));
    } catch (...) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw_error_already_set();
    }
}

// =============================================================================
//  get size
// =============================================================================
int
Projector::size (void) const
{
    return projections.size();
}

// =============================================================================
//  remove all layer
// =============================================================================
void
Projector::clear (void)
{
    projections.clear();
}

// =============================================================================
//  layers evaluation
// =============================================================================
void
Projector::connect (bool use_thread)
{
    if (use_thread) {
        boost::thread_group threads;
        for (unsigned int i=0; i<projections.size(); i++) {
            Projection::current = projections[i].get();
            threads.create_thread (&Projection::static_connect);
        }
        threads.join_all();    
    } else {
        for (unsigned int i=0; i<projections.size(); i++)
            projections[i]->connect();
    }

}

// =============================================================================
//    Boost wrapping code
// =============================================================================
void
Projector::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Projector> >();
    
    class_<Projector>("projector",
    "======================================================================\n"
    "\n"
    "A projector is a set of projections\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
    
        init<> (
        "__init__() -- initializes projector\n")
        )
 
        .def ("__getitem__", &Projector::get,
        "x.__getitem__ (y)  <==> x[y]\n\n"
        "Use of negative indices is supported.\n")

        .def ("__len__", &Projector::size,
        "__len__() -> integer -- return number of layers\n")
        
        .def ("append", &Projector::append,
        "append(projection) -- append layer to end\n")
        
        .def ("clear", &Projector::clear,
        "clear() -- remove all projection\n")
        
        .def ("connect", &Projector::connect,
        "connect(use_thread) -- instantiate all projections")
        ;
}

