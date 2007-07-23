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
#include "cube.h"
#include "background.h"
#include "colorbar.h"
#include "text.h"
#include "array.h"
#include "flat_surface.h"
#include "smooth_surface.h"
#include "cubic_surface.h"
#include "label.h"
#include "model.h"

BOOST_PYTHON_MODULE(_objects) {
    using namespace glpython::objects;
    
    Cube::python_export();
    Background::python_export();
    Colorbar::python_export();
    Text::python_export();
    Array::python_export();
    FlatSurface::python_export();
    SmoothSurface::python_export();
    CubicSurface::python_export();
    Label::python_export();
    Model::python_export();
}
