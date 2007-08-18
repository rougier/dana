//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#ifndef __DANA_PROJECTION_H__
#define __DANA_PROJECTION_H__

#include <boost/python.hpp>
#include "../core/layer.h"
#include "density/density.h"
#include "profile/profile.h"
#include "distance/distance.h"
#include "shape/shape.h"

using namespace boost::python;


namespace dana { namespace projection {

    // Forward declaration of shared pointers
    typedef boost::shared_ptr<class Projection>    ProjectionPtr;

    
    class Projection : public object {
    public:
        static Projection *current;
        
    public:
        Projection (void);
        virtual ~Projection (void);                
        void connect (object data=object());
        static void static_connect (void);

    public:
        shape::ShapePtr         shape;
        distance::DistancePtr   distance;
        profile::ProfilePtr     profile;
        density::DensityPtr     density;
        core::LayerPtr          src;
        core::LayerPtr          dst;
        bool                    self;

    public:
        static void boost (void);
    };

}} // namespace dana::projection

#endif

