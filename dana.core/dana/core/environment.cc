//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: environment.cc 275 2007-08-14 15:01:41Z rougier $

#include "environment.h"
#include "network.h"
#include "map.h"
#include "layer.h"

using namespace dana::core;

unsigned long Environment::epochs = 0;
Environment *Environment::env = 0;

//__________________________________________________________________Environment
Environment::Environment (void) : Object()
{}

//_________________________________________________________________~Environment
Environment::~Environment (void)
{}

//_____________________________________________________________________evaluate 
void
Environment::evaluate (void)
{}

//______________________________________________________________static_evaluate 
void
Environment::static_evaluate (void)
{
    if (env) {
        Environment *e = env;
        for (unsigned long i=0; i<epochs; i++) {
            e->evaluate ();
            e->barrier->wait();
        }
    }
}

//_______________________________________________________________________attach
void
Environment::attach (MapPtr map)
{
    std::vector<MapPtr>::iterator result;
    result = find (maps.begin(), maps.end(), map);
    if (result != maps.end())
        return;
    maps.push_back (MapPtr (map));
}

//________________________________________________________________python_export
void
Environment::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Environment> >();

    class_<Environment, bases <Object> > ("Environment",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "An Environment is able to modify map activities at each evaluation.   \n"
    "It does not depend on anything other than itself.                     \n" 
    "______________________________________________________________________\n",
    init< > ("__init__()"))
        
    .def ("attach", &Environment::attach,
          "attach(map)\n\nAttach a map to the environment\n")

    .def ("evaluate", &Environment::evaluate,
          "evaluate()\n\nEvaluate environment state\n")
    ;
}
