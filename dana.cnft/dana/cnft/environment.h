//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CFNT_ENVIRONMENT_H__
#define __DANA_CNFT_ENVIRONMENT_H__

#include <boost/python.hpp>
#include "core/environment.h"

using namespace boost::python;

namespace dana { namespace cnft {

    typedef boost::shared_ptr<class Environment> EnvironmentPtr;

    class Environment : public core::Environment {
        public:
            int     number;
            float	width;
            float	intensity;
            float	radius;
            float	dtheta;
            float	theta;
            float   noise;

        public:
            Environment (int number = 3,
                         float width = 0.1f,
                         float intensity = 1.5,
                         float radius = 0.70f,
                         float theta = 0.0f,
                         float dtheta = 1.0,
                         float noise = 0.0f);
            virtual ~Environment (void);
            virtual void evaluate  (void);
            virtual void gaussian (core::MapPtr map, core::LayerPtr layer,
                                   float center_x, float center_y,
                                   float width_x, float width_y,
                                   float intensity);
            static void python_export (void);
    };
}} // namespace dana::core

#endif
