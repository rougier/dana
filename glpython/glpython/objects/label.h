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

#ifndef __GLPYTHON_OBJECTS_LABEL_H__
#define __GLPYTHON_OBJECTS_LABEL_H__

#include <string>
#include "../core/object.h"
#include "../core/color.h"


namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Label> LabelPtr;
    
    class Label : public core::Object {
        public:
            std::string       text;
            core::ColorPtr    fg_color;
            core::ColorPtr    bg_color;
            core::ColorPtr    br_color;
            float             size;
            float             x,y,z1,z2;
            float             alpha;
            int               list;

        public:
            Label (std::string text = "Label",
                   tuple position = make_tuple (0,0,0,.5),
                   tuple fg_color = make_tuple (0,0,0,1),
                   tuple bg_color = make_tuple (1,1,1,1),
                   tuple br_color = make_tuple (0,0,0,1),
                   float size = 12.0,
                   float alpha = 1.0f,
                   std::string name = "label");
            virtual ~Label (void);

            virtual void        set_text        (std::string text);
            virtual std::string get_text        (void);
            virtual void        set_position    (tuple position);
            virtual tuple       get_position    (void);
            virtual void        set_fg_color    (tuple color);
            virtual tuple       get_fg_color    (void);
            virtual void        set_bg_color    (tuple color);
            virtual tuple       get_bg_color    (void);
            virtual void        set_br_color    (tuple color);
            virtual tuple       get_br_color    (void);
            virtual void        set_size        (float size);
            virtual float       get_size        (void);
            virtual void        set_alpha       (float alpha);
            virtual float       get_alpha       (void);

            virtual void        render          (void);
            static void         python_export   (void);
    };
}}

#endif
