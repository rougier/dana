/*
  DANA - Distributed Asynchronous Numerical Adaptive computing library
  Copyright (c) 2006,2007,2008 Nicolas P. Rougier

  This file is part of DANA.

  DANA is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free
  Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  DANA is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  You should have received a copy of the GNU General Public License
  along with DANA. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdlib>
#include "random.h"

using namespace dana::core;


Random::Random (void) : Object()
{
    set_seed (0);
}

Random::~Random (void)
{
}


void
Random::set_seed (unsigned int seed)
{
    this->seed = seed;
    srandom (seed);
}

unsigned int
Random::get_seed (void)
{
    return seed;
}


void
Random::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Random> >();
 
    class_<Random>("Random",
    "======================================================================\n"
    "\n"
    "A model gathers one to several networks and environments. Evaluation is\n"
    "done first on environments then on networks."
    "\n"
    "\n"
    "======================================================================\n",
        init<>(
        "__init__() -- initializes Random\n")
        )

        .add_property ("seed",
                       &Random::get_seed, &Random::set_seed,
                       "Seed that governs all random generators (c++ and python)")
        ;
}

