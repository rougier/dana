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


Model *Model::current_model = 0;

// ============================================================================
//  constructor
// ============================================================================
Model::Model (void) : Object()
{
    running = false;
    age = 0;
}

// ============================================================================
//  destructor
// ============================================================================
Model::~Model (void)
{}


// ============================================================================
//  append a new network
// ============================================================================
void
Model::append (NetworkPtr net)
{
    std::vector<NetworkPtr>::iterator result;
    result = find (networks.begin(), networks.end(), net);
    if (result != networks.end())
        return;

    networks.push_back (NetworkPtr(net));
}

// ============================================================================
//  append a new network
// ============================================================================
void
Model::append (EnvironmentPtr env)
{
    std::vector<EnvironmentPtr>::iterator result;
    result = find (environments.begin(), environments.end(), env);
    if (result != environments.end())
        return;

    environments.push_back (EnvironmentPtr(env));
}

// ============================================================================
//  Remove all networks and environments
// ============================================================================
void
Model::clear (void)
{
    networks.clear();
    environments.clear();
}

// ============================================================================
//  Start model evaluation for the given number of epochs
// ============================================================================
bool
Model::start (unsigned long n)
{
    if (running)
        return false;

	if (n > 0) {
		start_time = time;
		stop_time  = time + n;
	} else {
		start_time = 0;
		stop_time  = 0;
	}

	current_model = this;
	boost::thread T(&Model::entry_point);
	return true;
}

// ============================================================================
//  Entry point for threaded evaluation
// ============================================================================
void
Model::entry_point (void)
{
    Model *model = Model::current_model;

    model->running = true;
    bool go = true;
    
    while (go) {
        for (unsigned int i = 0; i < model->environments.size(); i++)
            model->environments[i]->evaluate ();
        for (unsigned int i = 0; i < model->networks.size(); i++)
            model->networks[i]->evaluate (1, false);
        model->time += 1;
        model->age++;

        if (((model->stop_time > 0) && (model->time >= model->stop_time)) ||
            (!model->running))
            go = false;
    }
    model->running = false;
}

// ============================================================================
//  Stop simulation
// ============================================================================
void
Model::stop (void)
{
    current_model->running = false;
}


// ============================================================================
//    Boost wrapping code
// ============================================================================
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(start_overloads, start, 0, 1)

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

        
        .def ("start", &Model::start, start_overloads (
                args("epochs"), "start(epochs = 0) -- start simulation\n"))
        .def ("stop",     &Model::stop,
              "stop() -- stop simulation\n")
        .def_readonly ("running", &Model::running)
        .def_readonly ("age", &Model::age)
        ;
}

