/*
  DANA - Distributed Asynchronous Numerical Adaptive computing library
  Copyright (c) 2006,2007,2008 Nicolas P. Rougier

  This file is part of DANA.

  DANA is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free
  Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  DANA is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  You should have received a copy of the GNU General Public License
  along with DANA. If not, see <http://www.gnu.org/licenses/>.
*/

#include "link.h"
#include "unit.h"

using namespace dana::core;

// _________________________________________________________________________Link
Link::Link ()
{}

// _________________________________________________________________________Link
Link::Link (UnitPtr source, float weight)
    : source (source), weight (weight)
{}

// ________________________________________________________________________~Link
Link::~Link (void)
{}

// ________________________________________________________________________write
int
Link::write (xmlTextWriterPtr writer)
{
    // <Link>
    xmlTextWriterStartElement (writer, BAD_CAST "Link");

    // weigth = ""
    xmlTextWriterWriteFormatAttribute (writer, BAD_CAST "weight", "%f", weight);

    // </Link>
    xmlTextWriterEndElement (writer);
    return 0;
}

// _________________________________________________________________________read
int
Link::read (xmlTextReaderPtr reader)
{
    std::istringstream iss;
    iss.clear();
    iss.str(read_attribute (reader, "weight"));
    iss >> weight;

    return 0;
}

// __________________________________________________________________________get
py::tuple const
Link::get (void)
{
    return py::make_tuple (source, weight);
}

// ___________________________________________________________________get_source
UnitPtr const
Link::get_source (void)
{
	return source;
}

// ___________________________________________________________________set_source
void
Link::set_source (UnitPtr source) 
{
    this->source = source;
}

// ___________________________________________________________________get_weight
float const
Link::get_weight (void)
{
	return weight;
}

// ___________________________________________________________________set_weight
void
Link::set_weight (float weight)
{
    this->weight = weight;
}

// ______________________________________________________________________compute
float
Link::compute (void)
{
    return source->potential * weight;
}

// ________________________________________________________________python_export
void
Link::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Link> >();
   
    class_<Link> ("Link",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A link describes the influence of a source over a target that owns the\n"
    "link.                                                                 \n"
    "______________________________________________________________________\n",

    init < UnitPtr, optional <float> > (
        (arg("source"),
         arg("weight") = 0.0f),
        "__init__ (source, weight=0)\n"))

    .add_property ("source",
                   &Link::get_source, &Link::set_source,
                   "source that feed the link")
    .add_property ("weight",
                   &Link::get_weight, &Link::set_weight,
                   "weight of the link")
     ;
}
