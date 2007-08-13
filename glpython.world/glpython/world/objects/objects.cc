//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <boost/python.hpp>
#include "disc.h"
#include "bar.h"
#include "box.h"

BOOST_PYTHON_MODULE(_objects) {
    using namespace glpython::world::objects;
    Disc::python_export();
    Bar::python_export();
    Box::python_export();
}
