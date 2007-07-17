//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "colormaps.h"

using namespace glpython::core;

ColormapPtr Colormaps::Ice        = ColormapPtr ( new Colormap ());
ColormapPtr Colormaps::Fire       = ColormapPtr ( new Colormap ());
ColormapPtr Colormaps::IceAndFire = ColormapPtr ( new Colormap ());
ColormapPtr Colormaps::Hot        = ColormapPtr ( new Colormap ());
ColormapPtr Colormaps::Gray       = ColormapPtr ( new Colormap ());
ColormapPtr Colormaps::Red        = ColormapPtr ( new Colormap ());
ColormapPtr Colormaps::Green      = ColormapPtr ( new Colormap ());
ColormapPtr Colormaps::Blue       = ColormapPtr ( new Colormap ());

ColormapPtr Colormaps::Default    = ColormapPtr (IceAndFire);

void
Colormaps::make (void)
{
    Colormaps::IceAndFire->clear();
    Colormaps::IceAndFire->append ( 0.00f, make_tuple (0.0f, 0.0f, 1.0f));
    Colormaps::IceAndFire->append ( 0.25f, make_tuple (0.5f, 0.5f, 1.0f));
    Colormaps::IceAndFire->append ( 0.50f, make_tuple (1.0f, 1.0f, 1.0f));
    Colormaps::IceAndFire->append ( 0.75f, make_tuple (1.0f, 1.0f, 0.0f));
    Colormaps::IceAndFire->append ( 1.00f, make_tuple (1.0f, 0.0f, 0.0f));

    Colormaps::Ice->clear();
    Colormaps::Ice->append (0.00f, make_tuple (0.0f, 0.0f, 1.0f));
    Colormaps::Ice->append (0.50f, make_tuple (0.5f, 0.5f, 1.0f));
    Colormaps::Ice->append (1.00f, make_tuple (1.0f, 1.0f, 1.0f));

    Colormaps::Fire->append (0.00f, make_tuple (1.0f, 1.0f, 1.0f));
    Colormaps::Fire->append (0.50f, make_tuple (1.0f, 1.0f, 0.0f));
    Colormaps::Fire->append (1.00f, make_tuple (1.0f, 0.0f, 0.0f));

    Colormaps::Hot->clear();
    Colormaps::Hot->append ( 0.00f, make_tuple (0.0f, 0.0f, 0.0f));
    Colormaps::Hot->append ( 0.33f, make_tuple (1.0f, 0.0f, 0.0f));
    Colormaps::Hot->append ( 0.66f, make_tuple (1.0f, 1.0f, 0.0f));
    Colormaps::Hot->append ( 1.00f, make_tuple (1.0f, 1.0f, 1.0f));

    Colormaps::Gray->clear();
    Colormaps::Gray->append ( 0.0f, make_tuple (0.0f, 0.00f, 0.00f));
    Colormaps::Gray->append ( 1.0f, make_tuple (1.0f, 1.00f, 1.00f));

    Colormaps::Red->clear();
    Colormaps::Red->append ( 0.0f, make_tuple (0.0f, 0.00f, 0.00f));
    Colormaps::Red->append ( 1.0f, make_tuple (1.0f, 0.00f, 0.00f));

    Colormaps::Green->clear();
    Colormaps::Green->append ( 0.0f, make_tuple (0.0f, 0.00f, 0.00f));
    Colormaps::Green->append ( 1.0f, make_tuple (0.0f, 1.00f, 0.00f));

    Colormaps::Blue->clear();
    Colormaps::Blue->append ( 0.0f, make_tuple (0.0f, 0.00f, 0.00f));
    Colormaps::Blue->append ( 1.0f, make_tuple (0.0f, 0.00f, 1.00f));

}

void
Colormaps::python_export(void) {
    using namespace boost::python; 

    Colormaps::make();
    scope().attr("CM_Default")      = Default;
    scope().attr("CM_IceAndFire")   = IceAndFire;
    scope().attr("CM_Ice")          = Ice;
    scope().attr("CM_Fire")         = Fire;
    scope().attr("CM_Hot")          = Hot;
    scope().attr("CM_Gray")         = Gray;
    scope().attr("CM_Grey")         = Gray;
    scope().attr("CM_Red")          = Red;
    scope().attr("CM_Green")        = Green;
    scope().attr("CM_Blue")         = Blue;
    
}
