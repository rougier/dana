//
// Copyright (C) 2007,2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "unit.h"
#include "learner.h"

BOOST_PYTHON_MODULE(_learn) {
    using namespace dana::learn;
    Unit::boost();
    Learner::boost();
}
