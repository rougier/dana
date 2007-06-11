//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: gl.cc 145 2007-05-10 14:18:42Z rougier $

#include <boost/python.hpp>
#include "array.h"
#include "array_bar.h"
#include "colormap.h"

BOOST_PYTHON_MODULE(_gl) {
    using namespace dana::gl;
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");
  
    Array::boost();
    ArrayBar::boost();
    Colormap::boost();
    Color::boost();    
}
