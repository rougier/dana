//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <algorithm>
#include <boost/thread/thread.hpp>
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
Model::evaluate (unsigned long epochs, bool use_thread)
{
    if (use_thread) {
        boost::thread_group threads;
        
        int n = 0;
        for (unsigned int i=0; i<networks.size(); i++)
            n += networks[i]->maps.size();
        n+= environments.size();
        
        barrier = new boost::barrier (n);
        Map::epochs = epochs;
        Environment::epochs = epochs;
                        
        for (unsigned int j=0; j<networks.size(); j++) {
            for (unsigned int i = 0; i<networks[j]->maps.size(); i++) {
                networks[j]->maps[i]->barrier = barrier;
                Map::map = networks[j]->maps[i].get();
                threads.create_thread (&Map::evaluate);
            }
        }
        for (unsigned int i=0; i<environments.size(); i++) {
            environments[i]->barrier = barrier;
            Environment::env = environments[i].get();
            threads.create_thread (&Environment::static_evaluate);
        }
        threads.join_all();
        delete barrier;
     } else {
        for (unsigned long j=0; j<epochs; j++) {
           for (unsigned int i = 0; i < environments.size(); i++)
                environments[i]->evaluate ();
           for (unsigned int i = 0; i < networks.size(); i++)
                networks[i]->evaluate (1, false);
        }
     }
}


// ============================================================================
//    Boost wrapping code
// ============================================================================
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(evaluate_overloads, evaluate, 0, 2)

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
    "A model gathers one to several networks and environments. Evaluation is\n"
    "done first on environments then on networks."
    "\n"
    "Attributes:\n"
    "-----------\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__() -- initializes Model\n")
        )
   
        .def ("append", append_net)
        .def ("append", append_env,
        "append(net) -- append net to end\n",        
        "append(env) -- append env to end\n")

        .def ("clear", &Model::clear,
        "clear() -- remove all networks and environments\n")
        
        .def ("evaluate",    &Model::evaluate,
        evaluate_overloads (args("n", "use_thread"), 
        "evaluate(n=1, use_thread=false) -- evaluate model for n epochs")
        )     
        ;
}

