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

#ifndef __GLPYTHON_OBJECTS_TEXT_H__
#define __GLPYTHON_OBJECTS_TEXT_H__

#include <string>
#include "../core/object.h"
#include "../core/color.h"


namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Text> TextPtr;
    
    class Text : public core::Object {
        public:
            std::string       text;
            std::string       alignment;
            float             orientation;
            core::ColorPtr    color;
            float             size;
            float             x,y;
            float             alpha;
            int               list;

        public:
            Text (std::string text = "Text",
                  tuple position = make_tuple (0.5,-.1),
                  tuple color = make_tuple (0,0,0),
                  float size = 24,      
                  std::string alignment = "center",
                  float orientation = 0.0f,
                  float alpha = 1.0f,
                  std::string name = "text");
            virtual ~Text (void);

            virtual void        set_text        (std::string text);
            virtual std::string get_text        (void);
            virtual void        set_position    (tuple position);
            virtual tuple       get_position    (void);
            virtual void        set_color       (tuple color);
            virtual tuple       get_color       (void);
            virtual void        set_size        (float size);
            virtual float       get_size        (void);
            virtual void        set_alignment   (std::string alignment);
            virtual std::string get_alignment   (void);
            virtual void        set_orientation (float orientation);
            virtual float       get_orientation (void);
            virtual void        set_alpha       (float alpha);
            virtual float       get_alpha       (void);

            virtual void        render          (void);
            static void         python_export   (void);
    };
}}

#endif
