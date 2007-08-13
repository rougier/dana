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
//   An oriented colored bar
// 
// ============================================================================

#ifndef __GLPYTHON_WORLD_OBJECTS_BAR_H__
#define __GLPYTHON_WORLD_OBJECTS_BAR_H__

#include "glpython/core/object.h"
#include <GL/glu.h>

namespace glpython { namespace world { namespace objects {

    typedef boost::shared_ptr<class Bar> BarPtr;
    
    class Bar : public core::Object {
    public:
        float height,length;
        float theta,phi;
        float x,y,z;
        boost::python::list color;

        public:
            Bar (std::string name = "Bar");
            virtual ~Bar (void);

            virtual void render (void);
            virtual void bar (void);

            static void python_export (void);
    };
}}}


#endif
