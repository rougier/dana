//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <boost/tuple/tuple.hpp>
#include <iostream>
#include <iomanip>
#include "color.h"

using namespace glpython::core;

//________________________________________________________________________Color
Color::Color (float r, float g, float b, float a, float v)
{
	set (RED,r);
	set (GREEN, g);
	set (BLUE, b);
	set (ALPHA, a);
	data[VALUE] = v;
}

//________________________________________________________________________Color
Color::Color (tuple channels)
{
    set (RED,   0);
	set (GREEN, 0);
	set (BLUE,  0);
	set (ALPHA, 1);
	data[VALUE] = 0;
 
    int l = extract< float > (channels.attr("__len__")());
 
    try { set (RED,   extract <float> (channels[0])()); } catch (...) {
        throw_error_already_set(); PyErr_Print();
    }
    try { set (GREEN, extract <float> (channels[1])()); } catch (...) {
        throw_error_already_set(); PyErr_Print();
    }
    try { set (BLUE,  extract <float> (channels[2])()); } catch (...) {
        throw_error_already_set(); PyErr_Print();
    }
    if (l > 3) {
        try { set (ALPHA, extract <float> (channels[3])()); } catch (...) {
            throw_error_already_set(); PyErr_Print();
        }
    }
}


//________________________________________________________________________Color
Color::Color (const Color &other)
{
    data[RED]   = other.data[RED];
    data[GREEN] = other.data[GREEN];
    data[BLUE]  = other.data[BLUE];
    data[ALPHA] = other.data[ALPHA];
    data[VALUE] = other.data[VALUE];
}

//_______________________________________________________________________~Color
Color::~Color (void)
{}

//______________________________________________________________________get/set
void
Color::set (int index, float v)
{
    if (v > 1.0f)      v = 1.0f;
    else if (v < 0.0f) v = 0.0f;
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


//_________________________________________________________________________repr
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

//____________________________________________________________________operator=
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

//____________________________________________________________________operator+
Color
Color::operator+ (const Color &other)
{
	return Color (data[RED]  +other.data[RED],
	              data[GREEN]+other.data[GREEN],
	              data[BLUE] +other.data[BLUE],
                 (data[ALPHA]+other.data[ALPHA])/2,
                 (data[VALUE]+other.data[VALUE])/2);
}

//____________________________________________________________________operator*
Color
Color::operator* (const float scale)
{
	return Color (data[RED]*scale,
	              data[GREEN]*scale,
	              data[BLUE]*scale,
	              data[ALPHA],
	              data[VALUE]);
}

//__________________________________________________________________________cmp
bool
Color::cmp (Color c1, Color c2)
{
    return c1.data[VALUE] < c2.data[VALUE];
}


//________________________________________________________________python_export
void
Color::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Color> >();

    class_<Color> ("Color",
    " ______________________________________________________________________\n"
    "                                                                       \n"
    " A color specifies a red, green, blue and alpha value in the standard  \n"
    " color model. Each value ranges from 0.0 to 1.0 and is represented by  \n"
    " a floating point value. The alpha value defines opacity and ranges    \n"
    " from 0.0 to 1.0, where 0.0 means that the color is fully transparent  \n"
    " and 1.0 that the color is fully opaque. Beside the raw RGBA values,   \n"
    " a color also stores an extra value that can be used elsewhere.        \n"
    "_______________________________________________________________________\n",

    init< optional <float,float,float,float,float> > (
        (arg("red")   = 0,
         arg("green") = 0,
         arg("blue")  = 0,
         arg("alpha") = 1,
         arg("value") = 1),
        "__init__ ( red, green, blue, alpha, value )\n"
        "__init__ ( (red, green, blue) )\n"
        "__init__ ( (red, green, blue, alpha) )\n"))
    .def (init<Color>())
    .def (init<tuple>())

    .add_property ("red",   &Color::get_red,   &Color::set_red)
    .add_property ("green", &Color::get_green, &Color::set_green)
    .add_property ("blue",  &Color::get_blue,  &Color::set_blue)
    .add_property ("alpha", &Color::get_alpha, &Color::set_alpha)
    .add_property ("value", &Color::get_value)

    .def ("__repr__", &Color::repr,
            "x.__repr__() <==> repr(x)")
    ;
}
