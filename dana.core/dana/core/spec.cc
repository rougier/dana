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
