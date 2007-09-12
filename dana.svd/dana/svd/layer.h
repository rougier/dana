//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_SVD_LAYER_H__
#define __DANA_SVD_LAYER_H__

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <vector>
#include <map>
#include "dana/core/object.h"
#include <iostream>

// To perform the Singular Value Decomposition for the optimized convolution
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>

#define SVD_CORE_LINKS 0
#define SVD_SHARED_LINKS 1
#define SVD_SVD_LINKS 2

using namespace boost::python;


namespace dana { namespace svd {

    class Layer : public core::Layer {

    public:
        std::vector< gsl_matrix * > vU;
        std::vector< gsl_matrix * > vV;
        std::vector< gsl_vector * > vS;
        
        std::vector< core::LayerPtr > sources_shared;
        std::vector< core::LayerPtr > sources_svd;
        
        std::vector< int > types;

        std::vector< gsl_matrix *> kernels_shared;
        std::vector< gsl_matrix *> kernels_svd;
        
        std::vector< std::vector < gsl_matrix * > * > v_tmp_svd;
        
        public:
            // life management
            // ================================================================
            Layer (void);
            virtual ~Layer (void);
            void clear(void);
            
            // activity management
            // ================================================================
            // Methods used by a unit to recover the original weights
            int is_source(core::LayerPtr src_layer, int &type);
            int get_type(int index_src);


            gsl_matrix * get_matrix_pass_one(int index_src,int index_rank);
            void get_V_col(int index_src,int rank,gsl_vector * conv_vert);
            float get_eigen(int index_src,int rank);
            int get_rank(int index_src);

            gsl_matrix * get_kernel(int index_src, int type);

            int connect(int rank,gsl_matrix * U, gsl_vector * S, gsl_matrix * V, core::LayerPtr src,gsl_matrix * kernel);
            int connect(core::LayerPtr src, gsl_matrix * kernel);

            virtual float        compute_dp (void);

            // python export
            // ================================================================
            static void          boost (void);
    };

}} // namespace dana::svd

#endif
