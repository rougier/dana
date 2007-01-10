//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "environment.h"
#include "network.h"
#include "map.h"
#include "layer.h"
#include "unit.h"
#include "link.h"
#include "spec.h"
#include "model.h"

BOOST_PYTHON_MODULE(_core) {
    using namespace dana::core;
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");
  
    Model::boost();
    Spec::boost();
    Environment::boost();
    Network::boost();
    Map::boost();
    Layer::boost();
    Unit::boost();
    Link::boost();
}
