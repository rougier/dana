//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_LINK_SVD_H__
#define __DANA_CORE_LINK_SVD_H__

#include <boost/python.hpp>
#include "dana/core/object.h"
#include "dana/core/unit.h"
#include "dana/core/link.h"
#include "dana/core/layer.h"
#include "dana/core/map.h"

// To perform the Singular Value Decomposition for the optimized convolution
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>

using namespace boost::python;

namespace dana { namespace svd {
        
    class Link : public core::Link {
    public:
        int rank;
        int index;
        int kernel_size;
        int type; 
        // Defined in svd::Layer.h
        // SVD_CORE_LINKS for core::Links
        // SVD_SHARED_LINKS for shared links
        // SVD_SVD_LINKS for SVD links

        core::LayerPtr src;
        core::UnitPtr dst;

    public:
        Link (void);
        virtual ~Link (void);

        void set_kernel(int index,core::LayerPtr src, core::UnitPtr dst);    
        void set_svd(int index,int rank, int kernel_size,core::LayerPtr src,core::UnitPtr dst);
        
        virtual float compute (void);
        float compute_shared();
        float compute_svd();
        
    public:
        static void	boost (void);
    };

}} // namespace dana::svd

#endif

