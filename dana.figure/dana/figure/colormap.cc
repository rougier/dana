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

#include <algorithm>
#include "colormap.h"
using namespace boost::python::numeric;
using namespace dana::figure;

Colormap::Colormap (void)
{
	samples.clear();
	colors.clear();
	resolution = 512;
	min = 0.0f;
	max = 1.0f;
}

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

Colormap::~Colormap (void)
{}

std::string
Colormap::repr (void)
{
    std::ostringstream ost;
    for (unsigned int i=0; i < colors.size(); i++)
        ost << colors[i].repr() << "\n";
    return ost.str();
}

unsigned int
Colormap::len (void)
{
    return colors.size();
}

void
Colormap::clear (void)
{
	samples.clear();
	colors.clear();
}

void
Colormap::append (float val, py::object col)
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
        c.data[RED] = py::extract< float >(col[0])();
        c.data[GREEN] = py::extract< float >(col[1])();
        c.data[BLUE] = py::extract< float >(col[2])();
    } catch (...) {
        PyErr_Print();
        return;        
    }

    int l = py::extract< float > (col.attr("__len__")());
    if (l >= 4 )
        c.data[ALPHA] = py::extract< float >(col[3])();
    else
        c.data[ALPHA] = 1.0f;
    c.data [VALUE] = val;
	colors.push_back (c);
	sort (colors.begin(), colors.end(), Color::cmp);
    sample();
}

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
        py::throw_error_already_set();
    }
    return Color(1,1,1,1);
}

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

void
Colormap::scale (float min, float max)
{
    if (max > min) {
        this->min = min;
        this->max = max;
    }
    sample();
}

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

void
Colormap::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Colormap> >();

    class_<Colormap>
    ("Colormap",
     " Colormap                                                              \n"
     "                                                                       \n"
     " ______________________________________________________________________\n"
     "                                                                       \n"
     " A colormap is a vector of several colors, each of them having a value \n"
     " between 0 and 1 that defines a range used for color interpolation.    \n"
     " Color lookup requests for an argument smaller than the minimum value  \n"
     " evaluate to the first colormap entry. Requests for an argument greater\n"
     " than the maximum value evaluate to the last entry.                    \n"
     "                                                                       \n"
     " Attributes:                                                           \n"
     "                                                                       \n"
     "     min -- minimum representable value                                \n"
     "     max -- maximum representable value                                \n"
     " ______________________________________________________________________\n",

    init< >("Create a new colormap"))

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
            "Return exact interpolated color for value.\n")

    .def ("scale", &Colormap::scale,
            "scale(min,max)\n\n"
            "Scale colormap to match given mix/max bounds.\n")
    
    .def ("__getitem__", &Colormap::get,
            "x.__getitem__ (y)  <==> x[y]\n")

    .def ("__len__", &Colormap::len,
            "__len__() -> integer")

    .def_readwrite ("min", &Colormap::min, "minimum representable value")
    .def_readwrite ("max", &Colormap::max, "maximum representable value")
        ;
}

