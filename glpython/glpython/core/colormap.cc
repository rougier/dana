//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <algorithm>
#include "colormap.h"

using namespace boost::python::numeric;
using namespace glpython::core;

//_____________________________________________________________________Colormap
Colormap::Colormap (void)
{
	samples.clear();
	colors.clear();
	resolution = 512;
	min = 0.0f;
	max = 1.0f;
}

//_____________________________________________________________________Colormap
Colormap::Colormap (const Colormap &other)
{
	samples.clear();
	colors.clear();
	resolution = other.resolution;
	for (unsigned int i=0; i<other.colors.size(); i++)
	    colors.push_back (Color (other.colors[i]));
	min = other.min;
	max = other.max;
    sample();
}

//____________________________________________________________________~Colormap
Colormap::~Colormap (void)
{}

//_________________________________________________________________________repr
std::string
Colormap::repr (void)
{
    std::ostringstream ost;
    for (unsigned int i=0; i < colors.size(); i++)
        ost << colors[i].repr() << "\n";
    return ost.str();
}

//__________________________________________________________________________len
unsigned int
Colormap::len (void)
{
    return colors.size();
}

//________________________________________________________________________clear
void
Colormap::clear (void)
{
	samples.clear();
	colors.clear();
}

//_______________________________________________________________________append
void
Colormap::append (float val, object col)
{
    if ((val < 0.0f) || (val > 1.0f)) {
        PyErr_SetString(PyExc_ValueError, "Value must be between 0 and 1");
        return;
    }

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

    int l = extract< float > (col.attr("__len__")());
    if (l >= 4 )
        c.data[ALPHA] = extract< float >(col[3])();
    else
        c.data[ALPHA] = 1.0f;
    c.data [VALUE] = val;
	colors.push_back (c);
	sort (colors.begin(), colors.end(), Color::cmp);
    sample();
}

//__________________________________________________________________________get
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
    return Color(1,1,1,1);
}

//________________________________________________________________________color
Color
Colormap::color (float value)
{
    if (samples.empty())
        return Color (1,1,1,1);
    if (value <= min)
        return colors[0];
    else if (value >= max)
        return colors[colors.size()-1];
    value = (value-min)/(max-min);
    int index = int(value*(samples.size()-1));
    return samples[index];
}

//__________________________________________________________________exact_color
Color
Colormap::exact_color (float value)
{
    value = (value-min)/(max-min);

	if (colors.empty())
        return Color(1,1,1,1);
	Color sup_color = colors[0];
	Color inf_color = colors[colors.size()-1];

	if (value <= min)
		return colors[0];
	else if (value >= max)
		return colors[colors.size()-1];
	else if (colors.size () == 1)
		return colors[0];
	
    for (unsigned int i=1; i<colors.size(); i++) {
        if (value < colors[i].data[VALUE]) {
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

//________________________________________________________________________scale
void
Colormap::scale (float min, float max)
{
    if (max > min) {
        this->min = min;
        this->max = max;
    }
    sample();
}

//_______________________________________________________________________sample
void
Colormap::sample (void)
{
    samples.clear();
    for (int i=0; i<=resolution; i++) {
        float v = min + (i/float(resolution)) * (max-min);
        Color c = exact_color (v);
        samples.push_back (c);
    }
}

//________________________________________________________________python_export
void
Colormap::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Colormap> >();

    class_<Colormap> ("Colormap",
    "=====================================================================\n"
    "                                                                     \n"
    " A colormap is a vector of several colors, each of them having a val-\n"
    " ue between 0 and 1 that defines a range used for color interpolation.\n"
    " Color lookup requests for an argument smaller than the minimum value\n"
    " evaluate to the first colormap entry. Requests for an argument grea-\n"
    " ter than the maximum value evaluate to the last entry.              \n"
    "                                                                     \n"
    "=====================================================================\n",
    init< >("__init__ ()\n"))

    .def ("__repr__", &Colormap::repr,
            "x.__repr__() <==> repr(x)")
        
    .def ("clear", &Colormap::clear,
            "clear()\n\n"
            "Reset colormap (no value defined)\n")
        
    .def ("append", &Colormap::append,
            "append (value, color)\n\n"
            "Append a new value using specified RGBA tuple color.\n")

    .def ("color", &Colormap::color,
            "color(value) -> RGBA tuple\n\n"
            "Return interpolated color for value (from samples).\n")
        
    .def ("exact_color", &Colormap::exact_color,
            "exact_color(value) -> RGBA tuple\n\n"
            "Return interpolated color for value.\n")

    .def ("scale", &Colormap::scale,
            "scale(min,max)\n\n"
            "Scale colormap to match given mix/max bounds.\n")
    
    .def ("__getitem__", &Colormap::get,
            "x.__getitem__ (y)  <==> x[y]\n")

    .def ("__len__", &Colormap::len,
            "__len__() -> integer")
        ;
}

