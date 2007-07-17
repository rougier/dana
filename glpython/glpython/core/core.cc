//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <boost/python.hpp>
#include "object.h"
#include "viewport.h"
#include "camera.h"
#include "observer.h"
#include "colormap.h"
#include "colormaps.h"
    
BOOST_PYTHON_MODULE(_core) {
    using namespace glpython::core;
    
    Object::python_export();
    Viewport::python_export();
    Camera::python_export();
    Observer::python_export();
    Color::python_export();
    Colormap::python_export();
    Colormaps::python_export();    
}
