//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <numarray/arrayobject.h>
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
    
    Model::boost();
    Spec::boost();
    Environment::boost();
    Network::boost();
    Map::boost();
    Layer::boost();
    Unit::boost();
    Link::boost();
}
