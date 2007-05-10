//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __DANA_GL_COLORMAP_H__
#define __DANA_GL_COLORMAP_H__

#include <vector>
#include <boost/python.hpp>

using namespace boost::python;


namespace dana { namespace gl {

    class Color {
        public:
            float data[5];
        public:
           Color (float r=0, float g=0, float b=0, float a=1, float v=0);
           ~Color (void);
           Color operator+ (const Color &other);
           Color operator* (const float scale);
    };

    class Colormap {
        public:
            std::vector<Color> colors;
            std::vector<Color> map;
            int samples;
            float sup, inf;

        public:
            Colormap (void);
            ~Colormap (void);
            void clear (void);
            void add (object color, float value);
            float *colorfv (float value);
            Color color (float value);

            // python export
            // ================================================================
            static void boost (void);
    };

}} // namespace dana::gl

#endif
