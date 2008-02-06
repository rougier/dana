// -----------------------------------------------------------------------------
// DANA 
// Copyright (C) 2006-2007  Nicolas P. Rougier
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// this program.  If not, see <http://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------

#include <boost/python.hpp>
#include "object.h"
#include "color.h"
#include "colormap.h"
#include "colormaps.h"
    
BOOST_PYTHON_MODULE(_figure) {

    py::docstring_options doc_options;
    doc_options.disable_signatures();

    dana::figure::Object::python_export();
    dana::figure::Color::python_export();
    dana::figure::Colormap::python_export();
    dana::figure::Colormaps::python_export();
}
