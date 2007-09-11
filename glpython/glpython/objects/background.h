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
//  Description:
// ----------------------------------------------------------------------------
//
// ============================================================================

#ifndef __GLPYTHON_OBJECTS_BACKGROUND_H__
#define __GLPYTHON_OBJECTS_BACKGROUND_H__

#include <string>
#include <GL/gl.h>
#include <boost/python.hpp>
#include "../core/object.h"
#include "../core/colormap.h"


namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Background> BackgroundPtr;
    
    class Background : public core::Object {
        public:
            core::ColormapPtr   cmap;
            std::string         name;
            std::string         orientation;
            float               alpha;
            int                 list;
        public:
            Background (core::ColormapPtr cmap,
                        std::string orientation = "vertical",
                        float alpha = 1.0f, 
                        std::string name = "Background");
            virtual ~Background (void);

            virtual void                set_cmap        (core::ColormapPtr cmap);
            virtual core::ColormapPtr   get_cmap        (void);
            virtual void                set_orientation (std::string orientation);
            virtual std::string         get_orientation (void);
            virtual void                set_alpha       (float alpha);
            virtual float               get_alpha       (void);

            virtual void    render (void);
            static void     python_export (void);
    };
}}

#endif
