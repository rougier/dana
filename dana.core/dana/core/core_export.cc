//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: core.cc 245 2007-07-19 10:10:31Z rougier $

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
