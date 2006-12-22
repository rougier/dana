//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <algorithm>
#include "model.h"
#include "network.h"
#include "environment.h"

using namespace dana::core;


// =============================================================================
//  constructor
// =============================================================================
Model::Model (void) : Object()
{}

// =============================================================================
//  destructor
// =============================================================================
Model::~Model (void)
{}


// =============================================================================
//  append a new network
// =============================================================================
void
Model::append (NetworkPtr net)
{
    std::vector<NetworkPtr>::iterator result;
    result = find (networks.begin(), networks.end(), net);
    if (result != networks.end())
        return;

    networks.push_back (NetworkPtr(net));
}

// =============================================================================
//  append a new network
// =============================================================================
void
Model::append (EnvironmentPtr env)
{
    std::vector<EnvironmentPtr>::iterator result;
    result = find (environments.begin(), environments.end(), env);
    if (result != environments.end())
        return;

    environments.push_back (EnvironmentPtr(env));
}

// =============================================================================
//  Remove all networks and environments
// =============================================================================
void
Model::clear (void)
{
    networks.clear();
    environments.clear();
}

// =============================================================================
//   evaluates all units potential and returns difference
// =============================================================================
void
Model::evaluate (unsigned long n)
{}


// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Model::boost (void)
{
    register_ptr_to_python< boost::shared_ptr<Model> >();
 
    // member function pointers for overloading
    void (Model::*append_net)(NetworkPtr)     = &Model::append;
    void (Model::*append_env)(EnvironmentPtr) = &Model::append;
 
    class_<Model>("Model",
    "======================================================================\n"
    "\n"
    "A model gathers one to several networks and environments\n"
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__() -- initializes Model\n")
        )
   
        .def ("append", append_net,
        "append(net) -- append net to end\n")
        
        .def ("append", append_env,
        "append(env) -- append env to end\n")

        .def ("clear", &Model::clear,
        "clear() -- remove all networks and environments\n")
                
        ;
}

