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

#include <boost/python.hpp>
#include "../core/object.h"
#include "vec4f.h"

using namespace boost::python;

namespace glpython { namespace objects {

    typedef boost::shared_ptr<class Model> ModelPtr;

    class Model : public core::Object {
        public:

        public:
            Model (std::string filename,
                   float alpha = 1.0f,
                   std::string name = "Model");
            virtual ~Model (void);

            virtual void initialize (void);
            virtual void render     (void);
            virtual void update     (void);

            static void python_export (void);
    };

}}

#endif
