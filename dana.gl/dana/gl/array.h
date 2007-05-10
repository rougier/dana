//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __DANA_GL_ARRAY_H__
#define __DANA_GL_ARRAY_H__

#include <string>
#include <FTGL/FTGLTextureFont.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "colormap.h"

using namespace boost::python;


namespace dana { namespace gl {
    class Array {
        public:
            std::string     name;       // Name to be displayed
            object          frame;      // [x,y,w,h] in normalized (0,1) units
            PyArrayObject * array;      // 1 or 2 dimension(s) numpy::array
            int             list;       // Display list for cube
            bool            active;     // Whether object is active
            bool            visible;    // Whether object is visible
            float *         data;       // Array data in float RGBA format
            unsigned int    tex;        // Texture id
            Colormap        cmap;       // Colormap

            static int      id_counter; // Id counter for unique identification
            int             id;         // unique identification

            object          select_callback, select_data;
            object          unselect_callback, unselect_data;
            FTFont *        font;
            int             fontsize;

        public:
            //  life management
            // ================================================================
            Array (object array,
                   object frame = make_tuple (0.0, 0.0, 1.0, 1.0),
                   std::string name = "", int fontsize = 16);
            virtual ~Array(void);

            //  object management
            // ================================================================
            void initialize (void);
            void render (void);
            void set_data (object array);
            void select (int primary, int secondary);
            void unselect (void);
            void connect (std::string event, object callback, object data);

            // python export
            // ================================================================
            static void boost (void);
    };

}} // namespace dana::gl

#endif
