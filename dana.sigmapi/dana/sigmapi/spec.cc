//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#include "spec.h"

using namespace dana::sigmapi;

//
// -----------------------------------------------------------------------------
Spec::Spec (void) : core::Spec()
{
	alpha	= 13;
	tau		= 0.75f;
	min_du	= 0.01;
	baseline= 0.0;
    lrate	= 0.0f;
    min_act	= 0.0f;
    max_act	= 1.0f;
}

//
// -----------------------------------------------------------------------------
Spec::~Spec (void)
{}


// ===================================================================
//  Boost wrapping code
// ===================================================================

void
Spec::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Spec> >();

    class_<Spec, bases<core::Spec> > ("Spec",
         "===================================================================\n"
         "\n"
         "A spec is an object describing a set of parameters. The base spec\n"
         "does not hold any parameter and must be derived for an object to\n"
         "hold any useful parameters.\n"
         "\n"
         "Attributes:\n"
         "-----------\n"
         "   tau:      Time constant\n"
         "   alpha:    Scaling factor\n"
         "   min_du:   Minimum difference of activity to stop evaluation\n"
         "   baseline: Base unit activation\n"
         "   lrate:    Learning rate\n"
         "   min_act:  Minimum unit activation\n"
         "   max_act:  Maximum unit activation\n"
        "\n"
        "===================================================================\n",
            init< > (
            "__init__() -- initializes spec\n")
        )
        
        .def_readwrite ("tau",     &Spec::tau)
        .def_readwrite ("alpha",   &Spec::alpha)
        .def_readwrite ("min_du",  &Spec::min_du)
        .def_readwrite ("baseline",&Spec::baseline)
        .def_readwrite ("lrate",   &Spec::lrate)
        .def_readwrite ("min_act", &Spec::min_act)
        .def_readwrite ("max_act", &Spec::max_act)
        ;
}
