//
// Copyright (C) 2006-2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <cstdlib>
#include "random.h"

using namespace dana::core;


// ============================================================================
//  constructor
// ============================================================================
Random::Random (void) : Object()
{
    set_seed (0);
}

// ============================================================================
//  destructor
// ============================================================================
Random::~Random (void)
{}


// ============================================================================
//  Set new seed
// ============================================================================
void
Random::set_seed (unsigned int seed)
{
    this->seed = seed;
    srandom (seed);
}

// ============================================================================
//  Get current seed
// ============================================================================
unsigned int
Random::get_seed (void)
{
    return seed;
}


// ============================================================================
//    Boost wrapping code
// ============================================================================

void
Random::boost (void)
{
    register_ptr_to_python< boost::shared_ptr<Random> >();
 
    class_<Random>("Random",
    "======================================================================\n"
    "\n"
    "A model gathers one to several networks and environments. Evaluation is\n"
    "done first on environments then on networks."
    "\n"
    "Attributes:\n"
    "-----------\n"
    " - seed : seed for a sequence of pseudo-random number"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__() -- initializes Random\n")
        )

        .add_property ("seed", &Random::get_seed, &Random::set_seed)
        ;
}

