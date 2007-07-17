//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __GLPYTHON_OBJECTS_CUBIC_SURFACE_H__
#define __GLPYTHON_OBJECTS_CUBIC_SURFACE_H__

#include <boost/python.hpp>
#include "array.h"
#include "vec4f.h"

using namespace boost::python;


namespace glpython { namespace objects {

    typedef boost::shared_ptr<class CubicSurface> CubicSurfacePtr;

    class CubicSurface : public Array {
        public:
            Vec4f *     vertices;
            int         nvertices;
            Vec4f *     colors;
            uint *      indices;
            int         nindices;
            float       zscale;

        public:
            CubicSurface (numeric::array X,
                     tuple frame = make_tuple (0,0,1,1),
                     core::ColormapPtr colormap = core::Colormaps::Default,
                     float zscale = 0.25f,
                     float alpha = 1.0f,
                     bool has_grid = true,
                     std::string name = "CubicSurface");
            virtual ~CubicSurface (void);

            virtual void initialize (void);
            virtual void render     (void);
            virtual void update     (void);

            static void python_export (void);
    };

}} // namespace glpython::objects

#endif
