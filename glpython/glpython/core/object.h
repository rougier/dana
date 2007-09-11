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
//  An Object is an abstraction of an OpenGL object that is renderable on
//  a Viewport.
//
//  Attributes:
//    name - name of the object
//    visible - whether to render object
//    active - whether to update object
// 
// ============================================================================

#ifndef __GLPYTHON_CORE_OBJECT_H__
#define __GLPYTHON_CORE_OBJECT_H__

#include <string>
#include <GL/gl.h>
#include <boost/python.hpp>


namespace glpython { namespace core {

    typedef boost::shared_ptr<class Object> ObjectPtr;
    
    class Object {
        public:
            std::string name;
            bool        visible;
            bool        active;
            bool        dirty;
            bool        is_ortho;
            int         depth;
            static int  id_counter;
            int         id;

        public:
            Object (std::string name = "Object");
            virtual ~Object (void);

            virtual std::string repr (void);
            virtual void        initialize (void);
            virtual void        render (void);
            virtual void        select (int selection = 0);            
            virtual void        update (void);

            static void python_export (void);
    };
}}

#endif
