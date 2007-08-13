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
#include "viewport.h"
#include "robot.h"
#include "observer.h"
#include "camera.h"

BOOST_PYTHON_MODULE(_core) {
    using namespace glpython::world::core;
    Viewport::python_export();
    Robot::python_export();    
    Observer::python_export();
    Camera::python_export();
}
