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
// A camera describes the view matrix that is used for rendering a scene.
//
// Attributes:
//    is_ortho - whether camera is in ortho mode
//    zoom - zoom factor
//    near,far - near and far planes
//    aperture - objective aperture
// 
// ============================================================================

#ifndef __GLPYTHON_CORE_CAMERA_H__
#define __GLPYTHON_CORE_CAMERA_H__

#include <string>
#include <boost/python.hpp>


namespace glpython { namespace core {

    typedef boost::shared_ptr<class Camera> CameraPtr;
    
    class Camera {
        public:
            std::string name;
            float       near, far, aperture, zoom;
            float       is_ortho;
            int         sx, sy, w, h;

        public:
            Camera (std::string name = "Camera");
            virtual ~Camera (void);

            virtual std::string repr (void);

            virtual void push (void);
            virtual void pop (void);
            
            virtual void select_event (int x, int y);

            static void python_export (void);
    };
}}

#endif
