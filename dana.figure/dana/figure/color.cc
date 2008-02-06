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

#include <iostream>
#include <iomanip>
#include "color.h"
using namespace dana::figure;

Color::Color (float r, float g, float b, float a, float v)
{
	set (RED,r);
	set (GREEN, g);
	set (BLUE, b);
	set (ALPHA, a);
	data[VALUE] = v;
}

Color::Color (py::tuple channels)
{
    set (RED,   0);
	set (GREEN, 0);
	set (BLUE,  0);
	set (ALPHA, 1);
	data[VALUE] = 0;
 
    int l = py::extract< float > (channels.attr("__len__")());
 
    try { set (RED, py::extract <float> (channels[0])()); } catch (...) {
        py::throw_error_already_set(); PyErr_Print();
    }
    try { set (GREEN, py::extract <float> (channels[1])()); } catch (...) {
        py::throw_error_already_set(); PyErr_Print();
    }
    try { set (BLUE,  py::extract <float> (channels[2])()); } catch (...) {
        py::throw_error_already_set(); PyErr_Print();
    }
    if (l > 3) {
        try { set (ALPHA, py::extract <float> (channels[3])()); } catch (...) {
            py::throw_error_already_set(); PyErr_Print();
        }
    }
}

Color::Color (const Color &other)
{
    data[RED]   = other.data[RED];
    data[GREEN] = other.data[GREEN];
    data[BLUE]  = other.data[BLUE];
    data[ALPHA] = other.data[ALPHA];
    data[VALUE] = other.data[VALUE];
}

Color::~Color (void)
{}

void
Color::set (int index, float v)
{
    if (v > 1.0f)
        v = 1.0f;
    else if (v < 0.0f)
        v = 0.0f;
    data[index] = v;
}

void  Color::set_red   (float v) { set (RED,  v); }
void  Color::set_green (float v) { set (BLUE, v); }
void  Color::set_blue  (float v) { set (GREEN,v); }
void  Color::set_alpha (float v) { set (ALPHA,v); }
float Color::get_red (void)      { return data[RED]; }
float Color::get_green (void)    { return data[GREEN]; }
float Color::get_blue (void)     { return data[BLUE]; }
float Color::get_alpha (void)    { return data[ALPHA]; }
float Color::get_value (void)    { return data[VALUE]; }


std::string
Color::repr (void)
{
    std::ostringstream ost;
    ost.setf (std::ios::showpoint);
    ost.setf (std::ios::fixed);
    ost << "[("  << std::setprecision(2)
        << data[RED]   << ", " << data[GREEN] << ", "
        << data[BLUE]  << ", " << data[ALPHA] << ") : ("
        << data[VALUE] << ")]"; 
    return ost.str();
}

Color
Color::operator= (const Color &other)
{
    if (this == &other)
        return *this;
	data[RED]   = other.data[RED];
	data[GREEN] = other.data[GREEN];
	data[BLUE]  = other.data[BLUE];
    data[ALPHA] = other.data[ALPHA];
    data[VALUE] = other.data[VALUE];
    return *this;
}

Color
Color::operator+ (const Color &other)
{
	return Color (data[RED]  +other.data[RED],
	              data[GREEN]+other.data[GREEN],
	              data[BLUE] +other.data[BLUE],
                 (data[ALPHA]+other.data[ALPHA])/2,
                 (data[VALUE]+other.data[VALUE])/2);
}

Color
Color::operator* (const float scale)
{
	return Color (data[RED]*scale,
	              data[GREEN]*scale,
	              data[BLUE]*scale,
	              data[ALPHA],
	              data[VALUE]);
}

bool
Color::cmp (Color c1, Color c2)
{
    return c1.data[VALUE] < c2.data[VALUE];
}


void
Color::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Color> >();

    class_<Color>
    ("Color",
     " Color                                                                 \n"
     "                                                                       \n"
     " ______________________________________________________________________\n"
     "                                                                       \n"
     " A color specifies a red, green, blue and alpha value in the standard  \n"
     " color model. Each value ranges from 0.0 to 1.0 and is represented by  \n"
     " a floating point value. The alpha value defines opacity and ranges    \n"
     " from 0.0 to 1.0, where 0.0 means that the color is fully transparent  \n"
     " and 1.0 that the color is fully opaque. Beside the raw RGBA values,   \n"
     " a color also stores an extra value that can be used elsewhere.        \n"
     "                                                                       \n"
     " Attributes:                                                           \n"
     "    red   -- normalized red component                                  \n"
     "    green -- normalized green component                                \n"
     "    blue  -- normalized blue component                                 \n"
     "    alpha -- normalized alpha component                                \n"
     "    value -- normalized arbitrary value                                \n"
     "_______________________________________________________________________\n",

    init< optional <float,float,float,float,float> > (
        (arg("red")   = 0.0,
         arg("green") = 0.0,
         arg("blue")  = 0.0,
         arg("alpha") = 1.0,
         arg("value") = 1.0),
      "Create a new colort                                                    \n"
      "                                                                       \n"
      "Function signature                                                     \n"
      "------------------                                                     \n"
      "                                                                       \n"
      "Color ( red,green,blue,alpha,value )                                   \n"
      "Color ( (red,green,blue) )                                             \n"
      "Color ( (red,greeb,blue,alpha) )                                       \n"        
      "                                                                       \n"
      "Keyword arguments                                                      \n"
      "-----------------                                                      \n"
      "                                                                       \n"
      "red -- normalized red component                                        \n"
      "       default: 0.0                                                    \n"
      "                                                                       \n"
      "green -- normalized green component                                    \n"
      "         default: 0.0                                                  \n"
      "                                                                       \n"
      "blue -- normalized blue component                                      \n"
      "        default: 0.0                                                   \n"
      "                                                                       \n"
      "alpha -- normalized alpha component                                    \n"
      "         default: 1.0                                                  \n"
      "                                                                       \n"
      "value -- normalized arbitrary value                                    \n"
      "         default: 1.0                                                  \n"
      "                                                                       \n"))
    .def (init<Color>())
    .def (init<tuple>())

    .add_property ("red",   &Color::get_red,   &Color::set_red,
                   "normalized red component")

    .add_property ("green", &Color::get_green, &Color::set_green,
                   "normalized green component")

    .add_property ("blue",  &Color::get_blue,  &Color::set_blue,
                   "normalized blue component")
        
    .add_property ("alpha", &Color::get_alpha, &Color::set_alpha,
                   "normalized alpha component")

    .add_property ("value", &Color::get_value,
                   "normalized arbitraty value")

    .def ("__repr__", &Color::repr,
            "x.__repr__() <==> repr(x)")
    ;
}
