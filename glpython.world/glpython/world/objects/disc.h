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
//   An oriented colored disc
// 
// ============================================================================

#ifndef __GLPYTHON_WORLD_OBJECTS_DISC_H__
#define __GLPYTHON_WORLD_OBJECTS_DISC_H__

#include "glpython/core/object.h"
#include <GL/glu.h>

namespace glpython { namespace world { namespace objects {

    typedef boost::shared_ptr<class Disc> DiscPtr;
    
    class Disc : public core::Object {
    public:
        float theta,phi;
        float x,y,z;
        float radius;
        boost::python::list color;

        public:
            Disc (std::string name = "Disc");
            virtual ~Disc (void);

            virtual void render (void);
            virtual void disc (void);

            static void python_export (void);
    };
}}}


#endif
