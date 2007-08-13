//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
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
//  This viewport inherites glpython::core::Viewport and provides a method
//  to enable freefly in the scene
//
// ============================================================================

#ifndef __GLPYTHON_WORLD_CORE_VIEWPORT_H__
#define __GLPYTHON_WORLD_CORE_VIEWPORT_H__

#include <vector>
#include "glpython/core/object.h"
#include "glpython/core/viewport.h"
#include "observer.h"

using namespace boost::python;


namespace glpython { namespace world { namespace core {

    class Viewport : public glpython::core::Viewport {
    private:
        bool button_pressed;
        double dx;
        double dy;
        
        public:
            Viewport (tuple size = make_tuple (1.0f, 1.0f),
                      tuple position = make_tuple(0.0f, 0.0f),
                      bool has_border = true,
                      bool is_ortho = false,
                      std::string name = "Viewport");

            virtual ~Viewport (void);

            // Save
            void save(char * filename);
            
            // Events methods
            virtual void   key_press_event (std::string key);
            virtual void   pointer_motion_event (int x, int y);

            // Overload for test
            virtual void render();
            virtual void   button_press_event (int button, int x, int y);
            virtual void   button_release_event (int button, int x, int y);


            static void    python_export (void);
    };
}}}


#endif
