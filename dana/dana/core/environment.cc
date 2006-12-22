//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#include "environment.h"
#include "network.h"
#include "map.h"
#include "layer.h"

using namespace dana::core;

//
// -----------------------------------------------------------------------------
Environment::Environment (void) : Object()
{}

//
// -----------------------------------------------------------------------------
Environment::~Environment (void)
{}

//
// -----------------------------------------------------------------------------
void
Environment::evaluate (void)
{}
 
// Attach some map
// -----------------------------------------------------------------------------
void
Environment::attach (MapPtr map)
{}

            
// ===================================================================
//  Boost wrapping code
// ===================================================================

void
Environment::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Environment> >();

    class_<Environment>
        ("Environment",
         "===================================================================\n"
         "\n"
         "An environment is an object that is able to modify map activation\n"
         "at each evaluation. An environment is said to be autonomous since\n"
         "its evaluation does not depend on anything else than itself.\n"
         "\n"
         "Attributes:\n"
         "-----------\n"
        "\n"
        "===================================================================\n",
            init< > (
            "__init__() -- initializes environment\n")
        )
        
        .def ("attach", &Environment::attach,
        "attach(map) -- Attach a map to the environment\n")
        ;
}
