//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#ifndef __DANA_SVD_PROJECTION_H__
#define __DANA_SVD_PROJECTION_H__

#include <boost/python.hpp>
#include "dana/core/layer.h"
#include "dana/projection/projection.h"
#include "dana/projection/density/density.h"
#include "dana/projection/profile/profile.h"
#include "dana/projection/distance/distance.h"
#include "dana/projection/shape/shape.h"

// To perform the Singular Value Decomposition for the optimized convolution
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>

using namespace boost::python;


namespace dana { namespace svd {

    // Forward declaration of shared pointers
    //typedef boost::shared_ptr<class Projection>    ProjectionPtr;

    
    class Projection : public projection::Projection {
    private:
        void shared_connect();
        void svd_simple_connect();

    public:
        //static Projection *current;
        int                     separable;
        
    public:
        Projection ();
        virtual ~Projection ();                
        void connect (object data=object());
        //static void static_connect (void);


    public:
        static void boost (void);
    };

}} // namespace dana::svd

#endif

