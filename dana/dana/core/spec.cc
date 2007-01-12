//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "spec.h"

using namespace dana::core;

//
// -----------------------------------------------------------------------------
Spec::Spec (void) : Object()
{}

//
// -----------------------------------------------------------------------------
Spec::~Spec (void)
{}


// ===================================================================
//  Boost wrapping code
// ===================================================================

void
Spec::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Spec> >();

    class_<Spec>("Spec",
    "======================================================================\n"
    "\n"
    "A spec is an object describing a set of parameters. The base spec\n"
    "does not hold any parameter and must be derived for an object to\n"
    "hold any useful parameters.\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
            init< > (
            "__init__() -- initializes spec\n")
        )

        ;
}
