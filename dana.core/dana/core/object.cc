//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: object.cc 241 2007-07-19 08:52:13Z rougier $

#include "object.h"

using namespace dana::core;
using namespace boost;

//________________________________________________________________runtime_error
void runtime_error (RuntimeError const &x)
{
    PyErr_SetString(PyExc_RuntimeError, x.what());
}

//_______________________________________________________________________Object
Object::Object (void)
{}

//______________________________________________________________________~Object
Object::~Object (void)
{}

//_________________________________________________________________________self
ObjectPtr
Object::myself (void) {
    if (_internal_weak_this.expired())
        throw RuntimeError("Shared pointer not available.");
    shared_ptr<Object> self = shared_from_this();
    assert(self != 0);
    return self;
}

//________________________________________________________________python_export
void
Object::python_export (void)
{
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Object> >();
    register_exception_translator<RuntimeError>(&::runtime_error);

    class_<Object> ("Object", 
    "______________________________________________________________________\n"
    "                                                                      \n"
    "Based class for any object of dana.                                   \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "   self -- Pointer on the underlying shared pointer if it exists.     \n"
    "______________________________________________________________________\n",
    
     init<>())
        .add_property ("self",
                       &Object::myself,
                       "underlying shared pointer (if it exists)")
        ;
}
