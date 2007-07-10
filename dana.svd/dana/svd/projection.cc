//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <iostream>
#include <math.h>
#include "core/map.h"
#include "core/unit.h"
#include "core/link.h"
#include "link.h"
#include "layer.h"
#include "projection.h"

#define cmp_threshold 1e-5

using namespace dana::svd;

//projection::Projection *Projection::current = 0;

dana::svd::Projection::Projection () : projection::Projection ()
{
    //self = true;
    separable = 0;
}

dana::svd::Projection::~Projection ()
{}


// =============================================================================
//  Connect methods
// =============================================================================

void
dana::svd::Projection::connect (object data)
{
    switch(separable)
        {
        case 0: 
            {   
                projection::Projection::connect(data);
            }
            break;
        case 1:
            {
                shared_connect();
            }
            break;
        case 2:
            { 
                svd_simple_connect();
            }
            break;
        default:
            printf("[ERROR] Unrecognized svd::Link type\n");
        } 
}

void 
dana::svd::Projection::shared_connect()
{
    int src_width  = src->map->width;
    int src_height = src->map->height;

    // Relative shape from a unit
    //std::vector<shape::vec2i> points;
    //shape->call (points, distance, 0, 0, src_width, src_height);
    //TODO : determine the width and height of the kernel
    // from the shape

    float fac = 2.0;

    int h_filt = int(fac*src_height);
    int w_filt = int(fac*src_width);
    
    gsl_matrix * kernel = gsl_matrix_alloc(h_filt,w_filt);

    // We first fill the kernel
    // considering that the dst unit is at (w_filt/2,h_filt/2)

    for(int i = 0 ; i < h_filt ; i++)
        {
            for(int j = 0 ; j < w_filt ; j++)
                {
                    float d = distance->call (0.5*w_filt/src_width,0.5*h_filt/src_height,float(j)/float(src_width),float(i)/float(src_height));
                    float w = profile->call(d);
                    gsl_matrix_set(kernel,i,j,w);
                }
        }

    // We connect the source layer to the destination layer
    int index = ((svd::Layer*)dst.get())->connect(src,kernel);
    
    for (int i=0; i<dst->size(); i++) {
        svd::Link * link = new svd::Link();
        link->set_kernel(index,src,dst->get(i));
        core::LinkPtr linkptr = core::LinkPtr(link);
        dst->get(i)->connect(linkptr);
    }
}

void
dana::svd::Projection::svd_simple_connect()
{
    int src_width  = src->map->width;
    int src_height = src->map->height;

    // Relative shape from a unit
    //std::vector<shape::vec2i> points;
    //shape->call (points, distance, 0, 0, src_width, src_height);
    //TODO : determine the width and height of the kernel
    // from the shape

    float fac = 2.0;

    int h_filt = int(fac*src_height);
    int w_filt = int(fac*src_width);
    
    gsl_matrix * kernel = gsl_matrix_alloc(h_filt,w_filt);
    gsl_matrix * U = gsl_matrix_alloc(h_filt,w_filt);    
    gsl_matrix * V = gsl_matrix_alloc(w_filt,w_filt);
    gsl_vector * S = gsl_vector_alloc(w_filt);
    gsl_vector * work = gsl_vector_alloc(w_filt);
    
    // We first fill the kernel
    // considering that the dst unit is at (w_filt/2,h_filt/2)

    for(int i = 0 ; i < h_filt ; i++)
        {
            for(int j = 0 ; j < w_filt ; j++)
                {
                    float d = distance->call (0.5*w_filt/src_width,0.5*h_filt/src_height,float(j)/float(src_width),float(i)/float(src_height));
                    float w = profile->call(d);
                    gsl_matrix_set(kernel,i,j,w);
                }
        }

    // We now have the kernel matrix, we compute the SVD
    
    // We must copy the kernel since gsl_linalg_SV_decomp
    // erase the provided kernel
    
    gsl_matrix_memcpy(U,kernel);
    
    // Decompose the matrix
    //gsl_linalg_SV_decomp_jacobi(U,V,S);
    
    gsl_linalg_SV_decomp (U,V,S,work);

    // and determine its rank
    int rank = 0;
    for(int i = 0 ; i < int(S->size) ; i++)
        {
            if(gsl_vector_get(S,i) >= cmp_threshold)
                {
                    rank++;
                }
        }
    
    /*
      TODO:ASSERT(dst.get() is a svd::Layer *)
      BOOST_ASSERT((boost::is_same<svd::Layer*, (layer.get())>::value)); 
    */

    // We connect the source layer to the destination layer
    int index = ((svd::Layer*)dst.get())->connect(rank,U,S,V,src,kernel);
    
    for (int i=0; i<dst->size(); i++) {
        svd::Link * link = new svd::Link();
        link->set_svd(index,rank,U->size2,src,dst->get(i));
        core::LinkPtr linkptr = core::LinkPtr(link);
        dst->get(i)->connect(linkptr);
    }
}


// =============================================================================
//
// =============================================================================
void
dana::svd::Projection::static_connect (void)
{
    if (current)
        current->connect();
}



// ============================================================================
//    Python export
// ============================================================================
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(connect_overloads, connect, 0, 1)

void
dana::svd::Projection::boost (void) {
    register_ptr_to_python< boost::shared_ptr<Projection> >();
    class_<Projection, bases<projection::Projection> >("projection",
                                                       "======================================================================\n"
                                                       "                                                                      \n"
                                                       " This class proposes two methods to connect a src and dst layers      \n"
                                                       " - The first one, shared_connect() defines shared weigths and put them\n"
                                                       " in the destination layer                                             \n"
                                                       " - The second, used by setting separable = 2, defines shared weights  \n"
                                                       "  that are computed using Singular Value Decomposition                \n"
                                                       " The syntax is exactly the same than for a projection::Projection     \n"
                                                       " except that you can choose between : \n"
                                                       "    - normal links (separable = 0) \n"
                                                       "    - shared links (separable = 1) \n"
                                                       "    - shared links computed with SVD (separable = 2) \n"
                                                       "\n"
                                                       "======================================================================\n",
                                                       init<> (
                                                               "init() -- initializes the projection\n"
                                                               )
                                                       )
        .def_readwrite ("separable", &Projection::separable)
        .def ("connect", &Projection::connect,connect_overloads (args("data"),"connect(data=None) -- instantiates the connection\n"))
        ;
}

