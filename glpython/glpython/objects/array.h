//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __GLPYTHON_OBJECTS_ARRAY_H__
#define __GLPYTHON_OBJECTS_ARRAY_H__

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "../core/object.h"
#include "../core/colormap.h"
#include "../core/colormaps.h"

using namespace boost::python;


namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Array> ArrayPtr;

    class Array : public core::Object {
        public:
            PyArrayObject *     array;
            float *             data;
            float               alpha;
            bool                has_grid;
            bool                has_border;
            tuple               frame;
            unsigned int        d0,d1;
            core::ColormapPtr   colormap;
            int                 sx, sy;
            object              select_callback, select_data;

        public:
            Array (numeric::array X,
                   tuple frame = make_tuple (0,0,1,1),
                   core::ColormapPtr colormap = core::Colormaps::Default,
                   float alpha = 1.0f,
                   bool has_grid = true,
                   bool has_border = true,                   
                   std::string name = "Array");
            virtual ~Array(void);

            virtual void           set_data (numeric::array X);
            virtual numeric::array get_data (void);
            virtual void           set_frame(tuple frame);
            virtual tuple          get_frame (void);
            virtual void           set_alpha (float value);
            virtual float          get_alpha (void);

            virtual void initialize (void);
            virtual void render     (void);
            virtual void select     (int selection = 0);
            virtual void update     (void);

            virtual void connect (std::string event,
                                  object callback,
                                  object data);

            static void python_export (void);
    };

}} // namespace glpython::objects

#endif
