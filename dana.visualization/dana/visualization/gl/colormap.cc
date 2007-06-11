//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: colormap.cc 145 2007-05-10 14:18:42Z rougier $

#include <algorithm>
#include <iostream>
#include <boost/python.hpp>
#include "colormap.h"

using namespace boost::python::numeric;
using namespace dana::gl;


// ============================================================================
Color::Color (float r, float g, float b, float a, float v)
{
	if (r > 1.0f) r = 1.0f;
	if (r < 0.0f) r = 0.0f;
	if (g > 1.0f) g = 1.0f;
	if (g < 0.0f) g = 0.0f;
	if (b > 1.0f) b = 1.0f;
	if (b < 0.0f) b = 0.0f;
	if (a > 1.0f) a = 1.0f;
	if (a < 0.0f) a = 0.0f;
	
	data[RED]   = r;
	data[GREEN] = g;
	data[BLUE]  = b;
	data[ALPHA] = a;
	data[VALUE] = v;
}

// ============================================================================
Color::Color (const Color &other)
{
    data[RED]   = other.data[RED];
    data[GREEN] = other.data[GREEN];
    data[BLUE]  = other.data[BLUE];
    data[ALPHA] = other.data[ALPHA];
    data[VALUE] = other.data[VALUE];
}

// ============================================================================
Color::~Color (void)
{
}

// ============================================================================
Color
Color::operator= (const Color &other)
{
    if (this == &other)
        return *this;
	return Color (other.data[RED],
	              other.data[GREEN],
	              other.data[BLUE],
                  other.data[ALPHA],
                  other.data[VALUE]);
}


// ============================================================================
Color
Color::operator+ (const Color &other)
{
	return Color (data[RED]  +other.data[RED],
	              data[GREEN]+other.data[GREEN],
	              data[BLUE] +other.data[BLUE],
                 (data[ALPHA]+other.data[ALPHA])/2,
                 (data[VALUE]+other.data[VALUE])/2);
}

// ============================================================================
Color
Color::operator* (const float scale)
{
	return Color (data[RED]*scale,
	              data[GREEN]*scale,
	              data[BLUE]*scale,
	              data[ALPHA],
	              data[VALUE]);
}

// ============================================================================
std::string
Color::repr (void)
{
    std::ostringstream ost;
    ost << "(("
        << data[RED]   << ", "
        << data[GREEN] << ", "
        << data[BLUE]  << ", "
        << data[ALPHA] << "), "
        << data[VALUE] << ")";      
    return ost.str();
}

// ============================================================================
bool
Color::cmp (Color c1, Color c2)
{
    return c1.data[VALUE] < c2.data[VALUE];
}


// ============================================================================
Colormap::Colormap (void)
{
	samples.clear();
	colors.clear();
	resolution = DEFAULT_RESOLUTION;
}

// ============================================================================
Colormap::~Colormap (void)
{
}

// ============================================================================
unsigned int
Colormap::len (void)
{
    return colors.size();
}

// ============================================================================
void
Colormap::clear (void)
{
	samples.clear();
	colors.clear();
}

// ============================================================================
void
Colormap::append (float val, object col)
{
	std::vector<Color>::iterator i;
	for (i=colors.begin(); i!= colors.end(); i++) {
        if (val == (*i).data[VALUE]) {
            colors.erase(i);
            break;
        }
    }

    Color c;
    try {
        c.data[RED] = extract< float >(col[0])();
        c.data[GREEN] = extract< float >(col[1])();
        c.data[BLUE] = extract< float >(col[2])();
    } catch (...) {
        PyErr_Print();
        return;        
    }

    try {
        c.data[ALPHA] = extract< float >(col[3])();
    } catch (...) {
        c.data[ALPHA] = 1.0f;
    }
    c.data [VALUE] = val;
	colors.push_back (c);
	sort (colors.begin(), colors.end(), Color::cmp);
    sample();
}

// ============================================================================
Color
Colormap::get (int index)
{
    int i = index;

    if (i < 0)
        i += colors.size();
    try {
        return colors.at(i);
    } catch (...) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw_error_already_set();
    }
    return Color(-1,-1,-1,-1);
}


// ============================================================================
Color
Colormap::color (float value)
{
    if (samples.empty())
        return Color (-1,-1,-1,-1);

    float inf = colors[0].data[VALUE];
    float sup = colors[colors.size()-1].data[VALUE];
    if (value < inf)
        value = inf;
    else if (value > sup)
        value = sup;
    int index = int((value-inf)/(sup-inf)*samples.size());
    return samples[index];
}

// ============================================================================
Color
Colormap::exact_color (float value)
{
    float inf = colors[0].data[VALUE];
    float sup = colors[colors.size()-1].data[VALUE];

	if (colors.empty())
        return Color(-1,-1,-1,-1);
	Color sup_color = colors[0];
	Color inf_color = colors[colors.size()-1];

	if (value < inf)
		return colors[0];
	else if (value > sup)
		return colors[colors.size()-1];
	else if (colors.size () == 1)
		return colors[0];
	
    for (unsigned int i=1; i<colors.size(); i++) {
        if (value > colors[i].data[VALUE]) {
            inf_color = colors[i-1];
            sup_color = colors[i];
            break;
        }
    }
	float r = fabs ((value-inf_color.data[VALUE])/
	                (sup_color.data[VALUE]-inf_color.data[VALUE]));
	Color c = sup_color*r + inf_color*(1.0f-r);
	c.data[VALUE] = value;
	return c;
}

// ============================================================================
void
Colormap::scale (float inf, float sup)
{
	std::vector<Color>::iterator i;
	for (i=colors.begin(); i!= colors.end(); i++)
        (*i).data[VALUE] = ((*i).data[VALUE] - inf) * (sup-inf);
    sample();
}

// ============================================================================
void
Colormap::sample (void)
{
    float inf = colors[0].data[VALUE];
    float sup = colors[colors.size()-1].data[VALUE];
    samples.clear();
    for (int i=0; i<=resolution; i++) {
        Color c = exact_color (inf+(i/float(resolution))*(sup-inf));
        samples.push_back (c);
    }
}

// ============================================================================
//   boost wrapping code
// ============================================================================
void
Colormap::boost (void) {

    using namespace boost::python;

    class_<Colormap> ("Colormap",
    "=====================================================================\n"
    "A colormap is a sequence of RGBA-tuples, where every tuple specifies \n"
    "color by a red, green and blue value in the RGB color model. Each va-\n"
    "lue ranges from 0.0 to 1.0 and is represented by a floating point va-\n"
    "lue. A fourth value, the so-called alpha value, defines opacity. It  \n"
    "also ranges from 0.0 to 1.0, where 0.0 means that the color is fully \n"
    "transparent, and 1.0 that the color is fully opaque. A colormap usua-\n"
    "lly stores 512 different RGBA-tuples, but other sizes are possible.  \n"
    "                                                                     \n"
    "Beside the raw RGBA values the colormap also stores one value per co-\n"
    "lor defining a value used for color interpolation. Color lookup requ-\n"
    "ests for an argument smaller than the minimum value evaluate to the  \n"
    "first colormap entry. Requests for an argument greater than the max- \n"
    "imum value evaluate to the last entry.                               \n"
    "=====================================================================\n",
        init< >(
              "__init__ ()\n"
              "\n"
              "Initialize colormap with no initial value.\n"))
        
        .def ("clear", &Colormap::clear,
              "clear()\n"
              "\n"
              "Reset colormap (no value defined)\n")
            
        .def ("append", &Colormap::append,
              "append (value, color)\n"
              "\n"
              "Append a new value using specified RGBA tuple color.\n")
            
        .def ("color", &Colormap::color,
              "color(value) -> RGBA tuple\n"
              "\n"
              "Return interpolated color for value.\n")
            
        .def ("scale", &Colormap::scale,
              "scale(min,max)\n"
              "\n"
              "Scale colormap to match given mix/max bounds.\n")
        
        .def ("__getitem__", &Colormap::get,
              "x.__getitem__ (y)  <==> x[y]\n"
              "\n"
              "Use of negative indices is supported.\n")

        .def ("__len__", &Colormap::len,
              "__len__() -> integer\n"
              "\n"
              "Return number of entries in the colormap.\n")
        ;
}

void
Color::boost (void) {

    using namespace boost::python;
    class_<Color> ("Color",
        "==================================================================\n"
        "\n"
        "==================================================================\n",
        init< optional <float,float,float,float,float> >
            ("__init__ () -- initialize color\n")
        )
        .def ("__repr__",   &Color::repr)
        ;
}
