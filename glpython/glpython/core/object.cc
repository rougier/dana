//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "object.h"

using namespace glpython::core;

int
Object::id_counter = 1;

//_______________________________________________________________________Object
Object::Object (std::string name) :
     name(name), visible(true), active(true), dirty(false), is_ortho(false), depth(50)
{
    id = id_counter++;
}

//______________________________________________________________________~Object
Object::~Object (void)
{}

//_________________________________________________________________________init
void
Object::initialize (void)
{}

//_______________________________________________________________________render
void
Object::render (void)
{}

//_______________________________________________________________________select
void
Object::select (int selection)
{}

//_______________________________________________________________________update
void
Object::update (void)
{
    dirty = true;
}

//_________________________________________________________________________repr
std::string
Object::repr (void)
{
    return name;
}

//________________________________________________________________python_export
void
Object::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Object> >();

    class_<Object>("Object",
    "======================================================================\n"
    "                                                                      \n"
    " An Object is an abstraction of an OpenGL object that is renderable   \n"
    " on a viewport.                                                       \n"
    "                                                                      \n"
    " Attributes:                                                          \n"
    "   name - name of the object                                          \n"
    "   visible - whether to render object                                 \n"
    "   active - whether to update object                                  \n"
    "                                                                      \n"
    "======================================================================\n",
     
    init< optional<std::string> >("__init__ ()"))
    
    .def_readwrite ("name",     &Object::name)
    .def_readwrite ("visible",  &Object::visible)
    .def_readwrite ("active",   &Object::active)
    .def_readwrite ("depth",    &Object::depth)
    
//    .def ("__repr__", &Object::repr,
//            "x.__repr__() <==> repr(x)")

    .def ("init", &Object::initialize,
            "init()\n\n"
            "Initialization of the object")

    .def ("render", &Object::render,
            "render()\n\n"
            "Rendering of the object.")

    .def ("update", &Object::render,
            "update()\n\n"
            "Updating of the object.")

    .def ("select", &Object::select,
            "select(id)\n\n"
            "Tell object that id has been selected.")

    ;       
}
