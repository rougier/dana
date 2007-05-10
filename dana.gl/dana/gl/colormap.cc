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
#include "colormap.h"

using namespace boost::python::numeric;
using namespace dana::gl;


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
	
	data[0] = r;
	data[1] = g;
	data[2] = b;
	data[3] = a;
	data[4] = v;
}

Color::~Color (void)
{}

Color
Color::operator+ (const Color &other)
{
	return Color (data[0]+other.data[0], data[1]+other.data[1], data[2]+other.data[2],
                  (data[3]+other.data[3])/2, (data[4]+other.data[4])/2);                   
}

Color
Color::operator* (const float scale)
{
	return Color (data[0]*scale, data[1]*scale, data[2]*scale,
                  data[3], data[4]);
}



Colormap::Colormap (void)
{
	map.clear();
	colors.clear();
	samples = 1024;
	inf = 0;
	sup = 0;
}

Colormap::~Colormap (void)
{}


void
Colormap::clear (void)
{
	map.clear();
	colors.clear();
}


void
Colormap::add (object col, float val)
{
	std::vector<Color>::iterator i;

	for (i=map.begin(); i!= map.end(); i++) {
        if (val == (*i).data[4])
            return;
        if (val > (*i).data[4])
            break;
    }

    Color c;
    try {
        c.data[0] = extract< float >(col[0])();
        c.data[1] = extract< float >(col[1])();
        c.data[2] = extract< float >(col[2])();
        c.data[3] = 1.0f;
        c.data[4] = val;
    } catch (...) {
        PyErr_Print();
        colors.clear();
        for (int i=0; i<=samples; i++) {
            inf = sup = 0;
            Color c = Color(1,1,1,1,0);
            colors.push_back (c);
        }
        return;
    }

	map.insert (i, c);
    if (map.size() == 1) {
        inf = sup = val;
    } else {
        if (val < inf)
            inf = val;
        else if (val > sup)
            sup = val;
    }

    colors.clear();
    for (int i=0; i<=samples; i++) {
        Color c = color (inf+(i/float(samples))*(sup-inf));
        colors.push_back (c);
    }
}


float *
Colormap::colorfv (float val)
{
    if (val < inf)
        val = inf;
    else if (val > sup)
        val = sup;
    int index = int((val-inf)/(sup-inf)*samples);
    return colors[index].data;
}


Color
Colormap::color (float value)
{
    Color c(1,1,1,1);

	if (map.empty())
        return c;
	Color sup_color = map[0];
	Color inf_color = map[map.size()-1];

	if (value < inf)
		return map[0];
	else if (value > sup)
		return map[map.size()-1];
	else if (map.size () == 1)
		return map[0];
	
    for (unsigned int i=1; i<map.size(); i++) {
        if (value > map[i].data[4]) {
            sup_color = map[i-1];
            inf_color = map[i];
            break;
        }
    }
	float r = fabs ((value-inf_color.data[4])/(sup_color.data[4]-inf_color.data[4]));
	c = sup_color*r + inf_color*(1.0f-r);
	return c;
}



// ============================================================================
//   boost wrapping code
// ============================================================================
void
Colormap::boost (void) {

    using namespace boost::python;

    class_<Colormap> ("Colormap",
    "======================================================================\n"
    "\n"
    "======================================================================\n",
        init< >(
        "__init__ () -- initialize colormap\n")
        )
        .def ("clear", &Colormap::clear)
        .def ("add",   &Colormap::add)
        ;
}
