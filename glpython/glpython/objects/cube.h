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
//   A red cube.
// 
// ============================================================================

#ifndef __GLPYTHON_OBJECTS_CUBE_H__
#define __GLPYTHON_OBJECTS_CUBE_H__

#include "../core/object.h"


namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Cube> CubePtr;
    
    class Cube : public core::Object {
        public:
            Cube (std::string name = "Cube");
            virtual ~Cube (void);

            virtual void render (void);
            virtual void cube (float x, float y, float z,
                               float dx, float dy, float dz);

            static void python_export (void);
    };
}}

#endif
