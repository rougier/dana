//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __GLPYTHON_WORLD_CORE_CAMERA_H__
#define __GLPYTHON_WORLD_CORE_CAMERA_H__

#include <string>
#include <boost/python.hpp>
#include "glpython/core/camera.h"

namespace glpython { namespace world { namespace core {

    class Camera : public glpython::core::Camera {

        public:
            Camera (std::string name = "Camera");
            virtual ~Camera (void);

            virtual void push (void);
            
            static void python_export (void);
    };
}}}

#endif
