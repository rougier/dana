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

#include "object.h"
#include "environment.h"
#include "network.h"
#include "map.h"
#include "layer.h"
#include "unit.h"
#include "link.h"
#include "spec.h"
#include "model.h"
#include "event.h"
#include "observer.h"
#include "observable.h"
#include "random.h"


BOOST_PYTHON_MODULE(_core) {
    using namespace dana::core;

    docstring_options doc_options;
    doc_options.disable_signatures();

    Object::python_export();
    Observable::python_export();
    Model::python_export();
    Spec::python_export();
    Environment::python_export();
    Network::python_export();
    Map::python_export();
    Layer::python_export();
    Unit::python_export();
    Link::python_export();
    Event::python_export();
    Observer::python_export();
    Random::boost();
}
