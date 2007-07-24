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
#include "../core/color.h"


using namespace boost::python;

namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Model> ModelPtr;

    class Model : public core::Object {
        public:
            Lib3dsFile *    file;        
            core::ColorPtr  color;
            float           alpha;

        public:
            Model (std::string filename,
                   tuple color = make_tuple (.75,.75,.75),
                   float alpha = 1.0f,
                   std::string name = "Model");
            virtual ~Model (void);
            virtual void render_node (Lib3dsNode *node, int mode = 0);
            virtual void render (void);
            static void python_export (void);
    };

}}

#endif
