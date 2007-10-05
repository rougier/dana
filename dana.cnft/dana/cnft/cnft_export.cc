//
// Copyright (C) 2007,2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "unit.h"
#include "spec.h"
#include "environment.h"

BOOST_PYTHON_MODULE(_cnft) {
    using namespace dana::cnft;
    docstring_options doc_options;
    doc_options.disable_signatures();

    Unit::boost();
    Spec::boost();
    Environment::python_export();
}
