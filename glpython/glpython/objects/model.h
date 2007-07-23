//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __GLPYTHON_OBJECTS_MODEL_H__
#define __GLPYTHON_OBJECTS_MODEL_H__

#include <string>
#include <vector>
#include <boost/python.hpp>
#include <lib3ds/file.h>
#include <lib3ds/camera.h>
#include <lib3ds/mesh.h>
#include <lib3ds/node.h>
#include <lib3ds/material.h>
#include <lib3ds/matrix.h>
#include <lib3ds/vector.h>
#include <lib3ds/light.h>
#include "../core/object.h"

using namespace boost::python;

namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Model> ModelPtr;

    typedef struct vertex {
        float x,y,z;
    };
    typedef struct color {
        float r,g,b;
    };
    typedef struct face {
        int v1, v2, v3, v4;
        int n1, n2, n3, n4;
    };
    typedef struct material {
        color   diffuse;
        color   ambient;
        color   specular;
    };
    typedef struct group  {
        std::string       name;
        material          mat;
        std::vector<face> faces;
    };


    class Model : public core::Object {
        public:
//            std::vector<vertex> vertices;
//            std::vector<vertex> normals;
//            std::vector<group>  groups;
            Lib3dsFile *file;

        public:
            Model (std::string filename, std::string name = "Model");
            virtual ~Model (void);
            virtual void render_node (Lib3dsNode *node);
            virtual void render (void);
            static void python_export (void);
    };

}}

#endif
