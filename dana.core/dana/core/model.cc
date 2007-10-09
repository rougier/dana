//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: model.cc 241 2007-07-19 08:52:13Z rougier $

#include <algorithm>
#include "model.h"
#include "network.h"
#include "environment.h"

using namespace boost;
using namespace dana::core;

// ________________________________________________________________________Model
Model::Model (void) : Object(), Observable (), age(0)
{}

// _______________________________________________________________________~Model
Model::~Model (void)
{}

// _______________________________________________________________________append
void
Model::append (NetworkPtr net)
{
    std::vector<NetworkPtr>::iterator result;
    result = find (networks.begin(), networks.end(), net);
    if (result != networks.end())
        return;
    
    networks.push_back (NetworkPtr(net));
    net->set_model (this);
}

// _______________________________________________________________________append
void
Model::append (EnvironmentPtr env)
{
    std::vector<EnvironmentPtr>::iterator result;
    result = find (environments.begin(), environments.end(), env);
    if (result != environments.end())
        return;

    environments.push_back (EnvironmentPtr(env));
    env->set_model (this);
}

// ________________________________________________________________________clear
void
Model::clear (void)
{
    networks.clear();
    environments.clear();
}

// ___________________________________________________________________compute_dp
void
Model::compute_dp (void)
{
    Py_BEGIN_ALLOW_THREADS
    using namespace std;
    vector<NetworkPtr>::const_iterator net;
    for(net=networks.begin(); net != networks.end(); net++) {
        (*net)->compute_dp();
    }
    Py_END_ALLOW_THREADS
}

// ___________________________________________________________________compute_dw
void
Model::compute_dw (void)
{
    Py_BEGIN_ALLOW_THREADS
    using namespace std;
    vector<NetworkPtr>::const_iterator net;
    for(net=networks.begin(); net != networks.end(); net++){
        (*net)->compute_dw();
    }
    Py_END_ALLOW_THREADS
}

// _____________________________________________________________________evaluate
void
Model::evaluate (unsigned long n)
{
    using namespace std;
    vector<EnvironmentPtr>::const_iterator env;
    EventEvaluatePtr event (new EventEvaluate());

    unsigned long i;
    for (i=0; i<n; i++){
        age++;
        for(env = environments.begin(); env != environments.end(); env++)
            (*env)->evaluate();        
        compute_dp();
        compute_dw();
        notify (event);
    }
}

// ______________________________________________________________________get_age
unsigned long int
Model::get_age (void)
{
    return age;
}

// _____________________________________________________________________get_spec
SpecPtr
Model::get_spec (void)
{
    return SpecPtr(spec);
}

// _____________________________________________________________________set_spec
void
Model::set_spec (SpecPtr spec)
{
    this->spec = SpecPtr(spec);
}

// _______________________________________________________________________export
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(start_overloads, start, 0, 1)

void
Model::python_export (void)
{
    register_ptr_to_python< boost::shared_ptr<Model> >();
 
    // member function pointers for overloading
    void (Model::*append_net)(NetworkPtr)     = &Model::append;
    void (Model::*append_env)(EnvironmentPtr) = &Model::append;

    //    PyEval_InitThread();
    
    class_<Model, bases <Object,Observable> >(
        "Model",
        "====================================================================\n"
        "\n"
        "A model gathers one to several networks and environments. Evaluation\n"
        "done first on environments then on networks."
        "\n"
        "===================================================================\n",
        init<>("__init__()\n"))

        .add_property ("spec",
                       &Model::get_spec, &Model::set_spec,
                       "Parameters of the model")

        .add_property ("age",
                       &Model::get_age,
                       "Age of the model")

        .def ("append", append_net)
        .def ("append", append_env,
              "append(net) -- append a network to the model\n",        
              "append(env) -- append an environment toe the model\n")

        .def ("clear", &Model::clear,
              "clear() -- remove all networks and environments\n")

        .def ("compute_dp",
              &Model::compute_dp,
              "compute_dp() -- calls compute_dp for each network\n")
        
        .def ("compute_dw",
              &Model::compute_dw,
              "compute_dw() -- calls compute_dw for each network\n")

        .def ("evaluate",
              &Model::evaluate,
              "evaluate(n) -- evaluates environments, compute_dp() and\n"
              "compute_dw() n times\n")
        ;
}

