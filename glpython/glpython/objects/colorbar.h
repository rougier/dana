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
// 
//
//
// ============================================================================

#ifndef __GLPYTHON_OBJECTS_COLORBAR_H__
#define __GLPYTHON_OBJECTS_COLORBAR_H__

#include "../core/object.h"
#include "../core/colormap.h"
#include "../core/colormaps.h"

namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Colorbar> ColorbarPtr;
    
    class Colorbar : public core::Object {
        public:
            core::ColormapPtr cmap;
            std::string       title;
            float             x,y;
            float             size;
            float             aspect;
            std::string       orientation;
            float             alpha;
            int               list;

        public:
            Colorbar (core::ColormapPtr cmap = core::Colormaps::Default,
                      std::string title = "",
                      float size = .8,
                      float aspect = 20.0,
                      tuple position = make_tuple (-0.12,.1),                      
                      std::string orientation = "vertical",
                      float alpha = 1.0f,
                      std::string name = "Colorbar");
            virtual ~Colorbar (void);

            virtual void                set_cmap        (core::ColormapPtr cmap);
            virtual core::ColormapPtr   get_cmap        (void);
            virtual void                set_title       (std::string title);
            virtual std::string         get_title       (void);
            virtual void                set_size        (float size);
            virtual float               get_size        (void);
            virtual void                set_aspect      (float aspect);
            virtual float               get_aspect      (void);            
            virtual void                set_position    (tuple position);
            virtual tuple               get_position    (void);
            virtual void                set_orientation (std::string orientation);
            virtual std::string         get_orientation (void);
            virtual void                set_alpha       (float alpha);
            virtual float               get_alpha       (void);

            virtual void    render          (void);
            static void     python_export   (void);
    };
}}

#endif
