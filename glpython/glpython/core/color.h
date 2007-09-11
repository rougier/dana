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
//   A color specifies a red, green, blue and alpha value in the RGBA color
//   model. Each value ranges from 0.0 to 1.0 and is represented by a floating
//   point value. The alpha value defines opacity. It also ranges from 0.0 to
//   1.0, where 0.0 means that the color is fully transparent, and 1.0 that the
//   color is fully opaque. Beside the raw RGBA values,  color also stores an
//   extra value that can be used inside a colormap.
//
// ============================================================================


#ifndef __GLPYTHON_CORE_COLOR_H__
#define __GLPYTHON_CORE_COLOR_H__

#include <boost/python.hpp>

using namespace boost::python;


namespace glpython { namespace core {

    const int RED   = 0;
    const int GREEN = 1;
    const int BLUE  = 2;
    const int ALPHA = 3;
    const int VALUE = 4;

    typedef boost::shared_ptr<class Color> ColorPtr;

    class Color {
        public:    
            float data[5];
            
        public:
            Color (float r=0, float g=0, float b=0, float a=1, float v=0);
            Color (tuple channels);
            Color (const Color &other);
            ~Color (void);

            void set        (int index, float r);
            void set_red    (float v);
            void set_green  (float v);
            void set_blue   (float v);
            void set_alpha  (float v);
            float get_red   (void);
            float get_green (void);
            float get_blue  (void);
            float get_alpha (void);
            float get_value (void);

            std::string repr (void);
            Color operator= (const Color &other);
            Color operator+ (const Color &other);
            Color operator* (const float scale);
            static bool cmp (Color c1, Color c2);

            static void python_export (void);
    };


}} // namespace dana::glpython::core

#endif
