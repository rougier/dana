//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: spec.cc 241 2007-07-19 08:52:13Z rougier $

#include "spec.h"

using namespace dana::core;

//_________________________________________________________________________Spec
Spec::Spec (void) : Object()
{}

//________________________________________________________________________~Spec
Spec::~Spec (void)
{}


//________________________________________________________________python_export
void
Spec::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Spec> >();

    class_<Spec, bases <Object> >("Spec",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A Spec is used as a set of parameters. The base spec does not hold any\n"
    "parameter and thus must be derived for an object to hold any useful   \n"
    "parameters.                                                           \n"
    "Attributes:                                                           \n"
    "______________________________________________________________________\n",
     init< > ( "__init__()"))

        ;
}
