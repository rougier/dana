//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "link.h"
#include "layer.h"

using namespace dana::svd;

#define max(a,b) (a>=b?a:b) 
#define min(a,b) (a<=b?a:b) 

//
// ----------------------------------------------------------------------------
Link::Link () : core::Link()
{}


//
// ----------------------------------------------------------------------------
Link::~Link (void)
{}

void Link::set_kernel(int index,core::LayerPtr src, core::UnitPtr dst)
{
    type = SVD_SHARED_LINKS;
    this->index = index;
    this->src = core::LayerPtr (src);
    this->source = core::UnitPtr(src->get(0));
    this->dst = core::UnitPtr(dst);
}

void Link::set_svd (int index, int rank, int kernel_size,core::LayerPtr src, core::UnitPtr dst)
{
    type = SVD_SVD_LINKS;
    this->index = index;
    this->rank = rank;
    this->kernel_size = kernel_size;
    this->src = src;
    this->source = src->get(0);
    this->dst = dst;
}

//
// ----------------------------------------------------------------------------
// // La deuxieme passe
float
Link::compute()
{

    switch(type)
        {
        case SVD_SHARED_LINKS:
            {
                // Shared links
                return compute_shared();
            }
            break;
        case SVD_SVD_LINKS:
            {
                // SVD links
                return compute_svd();
            }
            break;
        default:
            printf("[ERROR] Unrecognized link type \n");
        }
    return 0.0;
}

//
// ----------------------------------------------------------------------------
// Compute in the case of shared links
float
Link::compute_shared()
{
    float res = 0.0;
    int h_src = src->map->height;
    int w_src = src->map->width; 
    svd::Layer * dst_layer = ((svd::Layer *)(dst->layer));
    gsl_matrix * kernel = dst_layer->get_kernel(index,SVD_SHARED_LINKS);
    int h_filt = kernel->size1;
    int w_filt = kernel->size2;
    int pos_x = dst->get_x();
    int pos_y = dst->get_y();
    int i_min = max(0, int(pos_y - h_filt/2.0));
    int i_max = min(h_src, int(pos_y + h_filt/2.0));
    int j_min = max(0, int(pos_x - w_filt/2.0));
    int j_max = min(w_src, int(pos_x + w_filt/2.0));

    for(int i = i_min ; i < i_max ; i++)
        for(int j = j_min ; j < j_max ; j++) {
            int filt_x = int(j+w_filt/2.0 - pos_x) ;
            int filt_y = int(i+h_filt/2.0 - pos_y) ;
            res += (src->get(j,i)->potential)*(gsl_matrix_get(kernel,filt_y,filt_x));
        }
    
    return res;
}

//
// ----------------------------------------------------------------------------
// Compute in the case of Singular Value Decomposition
float
Link::compute_svd()
{
    float res = 0.0;
    gsl_matrix * dst_1D_tmp;
    gsl_vector * conv_vert = gsl_vector_alloc(kernel_size);
    
    for(int i = 0 ; i < rank ; i++)
        {
            svd::Layer * dst_layer = ((svd::Layer *)(dst->layer));

            dst_1D_tmp = dst_layer->get_matrix_pass_one(index,i);
            dst_layer->get_V_col(index,i,conv_vert);

            int w_filt = conv_vert->size;

            int k;
            int w_src = dst_1D_tmp->size2;
            
            int i_dst = dst->get_y();
            int j_dst = dst->get_x();
            int k_min = max(int(w_filt/2.0 - j_dst), 0);
            int k_max = min(int(w_src + w_filt/2.0 - j_dst), w_filt);
            for(k = k_min ; k < k_max ; k++)
                res += gsl_matrix_get (dst_1D_tmp, i_dst, int(j_dst+ k - w_filt/2.0))*gsl_vector_get (conv_vert, k);
        }
    
    gsl_vector_free(conv_vert);
    return res;
}

// ===================================================================
//  Boost wrapping code
// ===================================================================
void
Link::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Link> >();
   
    class_<Link, bases<core::Link> > ("Link",
                                      "======================================================================\n"
                                      "                                                                      \n"
                                      "A link describes the influence of a source over a target and is owned \n"
                                      "by the target.                                                        \n"
                                      " This class of links consider two types of links :                    \n"
                                      "    - shared links (type=SVD_SHARED_LINKS)                            \n"
                                      "    - shared links computed with SVD (type = SVD_SVD_LINKS)           \n"
                                      "                                                                      \n"
                                      "======================================================================\n",
        
                                      init< > ()
                                      )
        ;
}
