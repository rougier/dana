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
//  A colormap is a vector of several colors, each of them having a value bet-
//  ween 0 and 1 inclusive that defines a range used for color interpolation.
//  Color lookup requests for an argument smaller than the minimum value eva-
//  luate to the first colormap entry. Requests for an argument greater than
//  the maximum value evaluate to the last entry.
//
// ============================================================================


#ifndef __GLPYTHON_CORE_COLORMAP_H__
#define __GLPYTHON_CORE_COLORMAP_H__

#include <vector>
#include <boost/python.hpp>
#include "color.h"

using namespace boost::python;


namespace glpython { namespace core {

    typedef boost::shared_ptr<class Colormap> ColormapPtr;

    class Colormap {
        public:
            std::vector<Color>  colors;
            std::vector<Color>  samples;
            float               min, max;
            int                 resolution;

        public:
            Colormap                 (void);
            Colormap                 (const Colormap &other);            
            ~Colormap                (void);

            std::string repr         (void);
            
            unsigned int len         (void);
            void         clear       (void);
            void         append      (float value, object color);
            Color        get         (int index);
            Color        color       (float value);
            Color        exact_color (float value);            
            void         scale       (float inf, float sup);
            void         sample      (void);

            static void  python_export (void);
    };

}} // namespace dana::glpython::core

#endif
