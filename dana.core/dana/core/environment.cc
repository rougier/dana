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

#include "environment.h"
#include "model.h"
#include "network.h"
#include "map.h"
#include "layer.h"

using namespace dana::core;

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

// ____________________________________________________________________get_model
Model *
Environment::get_model (void)
{
    return model;
}

// ____________________________________________________________________set_model
void
Environment::set_model (class Model *model)
{
    this->model = model;
}

// _____________________________________________________________________get_spec
SpecPtr
Environment::get_spec (void)
{
    if ((spec == SpecPtr()) && (model))
        return model->get_spec();
    return SpecPtr(spec);

}

// _____________________________________________________________________set_spec
void
Environment::set_spec (SpecPtr spec)
{
    this->spec = SpecPtr(spec);
}


//________________________________________________________________python_export
void
Environment::python_export (void)
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
