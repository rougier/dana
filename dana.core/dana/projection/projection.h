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
#include "../core/object.h"
#include "../core/layer.h"
#include "density/density.h"
#include "profile/profile.h"
#include "distance/distance.h"
#include "shape/shape.h"

using namespace boost::python;


namespace dana { namespace projection {

    // Forward declaration of shared pointers
    typedef boost::shared_ptr<class Projection>    ProjectionPtr;

    
    class Projection : public core::Object {
        // ___________________________________________________________attributes
    public:
        shape::ShapePtr         shape;
        distance::DistancePtr   distance;
        profile::ProfilePtr     profile;
        density::DensityPtr     density;
        core::LayerPtr          src;
        core::LayerPtr          dst;
        bool                    self_connect;

    public:
        // _________________________________________________________________life
        Projection (void);
        virtual ~Projection (void);

        // _________________________________________________________________main
        void connect (object data=object());

        // ______________________________________________________________get/set
        virtual shape::ShapePtr       get_shape (void);
        virtual void                  set_shape (shape::ShapePtr shape);
        virtual profile::ProfilePtr   get_profile (void);
        virtual void                  set_profile (profile::ProfilePtr profile);
        virtual density::DensityPtr   get_density (void);
        virtual void                  set_density (density::DensityPtr density);
        virtual distance::DistancePtr get_distance (void);
        virtual void                  set_distance (distance::DistancePtr distance);
        virtual core::LayerPtr        get_src (void);
        virtual void                  set_src (core::LayerPtr src);
        virtual core::LayerPtr        get_dst (void);
        virtual void                  set_dst (core::LayerPtr dst);
        
        // _______________________________________________________________export
        static void python_export (void);
    };

}} // namespace dana::projection

#endif

