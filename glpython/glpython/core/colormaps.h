//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

// ============================================================================
//
//  Description:
//  ---------------------------------------------------------------------------
//
//  Some standard colormaps
//
// ============================================================================


#ifndef __GLPYTHON_CORE_COLORMAPS_H__
#define __GLPYTHON_CORE_COLORMAPS_H__

#include "colormap.h"

namespace glpython { namespace core {

    class Colormaps {
        public:
            static ColormapPtr Default;
            static ColormapPtr Ice;
            static ColormapPtr Fire;
            static ColormapPtr IceAndFire;
            static ColormapPtr Hot;
            static ColormapPtr Gray;
            static ColormapPtr Red;
            static ColormapPtr Green;
            static ColormapPtr Blue;

            static void make (void);
            static void python_export (void);
    };

}} // namespace glpython::core

#endif
