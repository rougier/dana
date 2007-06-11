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
//  A colormap is a sequence of RGBA-tuples, where every tuple specifies a
//  color by a red, green and blue value in the RGB color model. Each value
//  ranges from 0.0 to 1.0 and is represented by a floating point value. A
//  fourth value, the so-called alpha value, defines opacity. It also ranges
//  from 0.0 to 1.0, where 0.0 means that the color is fully transparent, and
//  1.0 that the color is fully opaque. A colormap usually stores 512 different
//  RGBA-tuples, but other sizes are possible too.
//
//  Beside the raw RGBA values the colormap also stores one value per color,
//  defining a range used for color interpolation. Color lookup requests for an
//  argument smaller than the minimum value evaluate to the first colormap
//  entry. Requests for an argument greater than the maximum value evaluate to
//  the last entry.
//
// ============================================================================


#ifndef __DANA_VISUALIZATION_GL_COLORMAP_H__
#define __DANA_VISUALIZATION_GL_COLORMAP_H__

#include <vector>
#include <boost/python.hpp>


using namespace boost::python;


namespace dana { namespace gl {

    //  Constants
    // ========================================================================
    const int RED   = 0;
    const int GREEN = 1;
    const int BLUE  = 2;
    const int ALPHA = 3;
    const int VALUE = 4;
    const int DEFAULT_SAMPLE_NUMBER = 512;


    // Color
    // ========================================================================
    class Color {
        public:    
            float data[5];
        public:
            Color (float r=0, float g=0, float b=0, float a=1, float v=0);
            Color (const Color &other);
            ~Color (void);

            Color operator= (const Color &other);
            Color operator+ (const Color &other);
            Color operator* (const float scale);
            std::string repr (void);
            static bool cmp (Color c1, Color c2);
            static void boost (void);
    };

    // Colormap
    // ========================================================================
    class Colormap {
        public:
            std::vector<Color>  colors;
            std::vector<Color>  samples;
            int                 resolution;

        public:
            Colormap (void);
            ~Colormap (void);

            unsigned int size(void);
            void         clear (void);
            void         add (float value, object color);
            Color        get (int index);
            Color        color (float value);
            Color        exact_color (float value);            

            void         scale (float inf, float sup);
            void         sample (int n = DEFAULT_SAMPLE_NUMBER);

            static void boost (void);
    };

}} // namespace dana::visualization::gl

#endif
