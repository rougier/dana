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
#include <numpy/arrayobject.h>
#include "projection.h"
#include "link.h"
#include "layer.h"
#include "unit.h"

BOOST_PYTHON_MODULE(_svd) {
    using namespace dana::svd;
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");
    Link::boost();
    Layer::boost();
    dana::svd::Projection::boost();
    Unit::boost();
}
