//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <algorithm>
#include "core/map.h"
#include "core/layer.h"
#include "layer.h"
#include "core/unit.h"

using namespace dana::svd;

#define max(a,b) (a>=b?a:b) 
#define min(a,b) (a<=b?a:b) 

// ============================================================================
//  constructor
// ============================================================================
Layer::Layer (void) : core::Layer()
{}

// ============================================================================
//  destructor
// ============================================================================
Layer::~Layer (void)
{
    unsigned int i;
    unsigned int j;
    
    for(i = 0 ; i < vU.size() ; i++)
        gsl_matrix_free(vU[i]);
    for(i = 0 ; i < vV.size() ; i++)
        gsl_matrix_free(vV[i]);    
    for(i = 0 ; i < vS.size() ; i++)
        gsl_vector_free(vS[i]);
    
    for(i = 0 ; i < v_tmp_svd.size() ; i++)
        for(j = 0 ; j < (v_tmp_svd[i])->size() ; j++)
            gsl_matrix_free((*(v_tmp_svd[i]))[j]);
}

// ============================================================================
//  Set to zero the potentials of all units
// ============================================================================
void
Layer::clear (void)
{
    for(unsigned i = 0 ; i < v_tmp_svd.size() ; i++)
        for(unsigned j = 0 ; j < (v_tmp_svd[i])->size() ; j++)
            gsl_matrix_set_zero((*(v_tmp_svd[i]))[j]);    

    core::Layer::clear();
}


/// Methods used by a unit to recover the original weights

int
Layer::is_source(core::LayerPtr src_layer, int &type)
{
    for(unsigned int i = 0 ; i < sources_shared.size() ; i++)
        {
            if(sources_shared[i] == src_layer)
                {
                    type = SVD_SHARED_LINKS;
                    return i;
                }
        }    
    for(unsigned int i = 0 ; i < sources_svd.size() ; i++)
        {
            if(sources_svd[i] == src_layer)
                {
                    type = SVD_SVD_LINKS;
                    return i;
                }   
        }       
    type = SVD_CORE_LINKS;
    return -1;
}

int 
Layer::get_rank(int index_src)
{
    return (v_tmp_svd[index_src])->size();
}

// ============================================================================
//   evaluates all units potential and returns difference
// ============================================================================

gsl_matrix *
Layer::get_matrix_pass_one(int index_src,int index_rank)
{
    return (*(v_tmp_svd[index_src]))[index_rank];
}

void
Layer::get_V_col(int index_src,int rank,gsl_vector * conv_vert)
{
    gsl_matrix_get_col(conv_vert,vV[index_src],rank);

}

float
Layer::get_eigen(int index_src,int rank)
{
    return gsl_vector_get(vS[index_src],rank);
}

gsl_matrix *
Layer::get_kernel(int index_src, int type)
{
    switch(type)
        {
        case SVD_CORE_LINKS:
            {
                printf("[ERROR] Layer::get_kernel :  Cannot access to kernel for linktype SVD_CORE_LINKS\n");
            }
            break;
        case SVD_SHARED_LINKS:
            {
                return kernels_shared[index_src];
            }
            break;
        case SVD_SVD_LINKS:
            {
                return kernels_svd[index_src];
            }
            break;
        default:
            printf("[ERROR] Layer::get_kernel : Unrecognized link type  \n");
        }
    return 0;
}

float
Layer::compute_dp (void)
{
    float d = 0.0;
    float du = 0.0;
    
    int index = 0;
    gsl_vector * conv_horiz;

    gsl_matrix * U;
    gsl_vector * S;
    gsl_matrix * V;
    gsl_matrix * dst_1D_tmp;
    int h_src,w_src,h_filt;
    float eigen_value,temp;
    
    // First pass, performed only for svd links

    for(unsigned int i = 0 ; i < sources_svd.size() ; i ++)
        {
            core::LayerPtr src = sources_svd[i];
            U = vU[i];
            S = vS[i];
            V = vV[i];
            
            h_src = src->get_map()->height;
            w_src = src->get_map()->width;
            h_filt = U->size1;
            
            conv_horiz = gsl_vector_alloc(U->size1); 
            for(int r = 0 ; r < int((v_tmp_svd[i])->size()) ; r++)
                {
                    dst_1D_tmp = (*(v_tmp_svd[i]))[r];
                    gsl_matrix_set_zero(dst_1D_tmp);
                    gsl_matrix_get_col(conv_horiz,U,r);
                    eigen_value = gsl_vector_get(S,r);
                    
                    for (int j = 0 ; j < h_src ; j++)
                        for (int k = 0 ; k  < w_src ; k++)
                            {
                                temp = 0.0;
                                int l_min = max(int(h_filt/2.0 - j) , 0);
                                int l_max = min(h_filt,int(h_src - j + h_filt/2.0));
                                for(int l = l_min ; l < l_max ; l++)
                                    temp += src->get(k,int(j+l-h_filt/2.0))->potential * gsl_vector_get(conv_horiz,l);
                                gsl_matrix_set(dst_1D_tmp,j,k,eigen_value * temp);
                            }
                }
            gsl_vector_free(conv_horiz);
        }

    // Second pass
    for (unsigned int m = 0; m< units.size(); m++) {
        index = map->shuffles[map->shuffle_index][m];
        du = units[index]->compute_dp();    
        int ux = units[index]->get_x();
        int uy = units[index]->get_y();
        for(unsigned int i = 0 ; i < sources_svd.size() ; i++)
            {
                // We determine which links are lateral
                if(sources_svd[i].get() == this)
                    {
                        // We then update the matrix of the first pass
                        U = vU[i];
                        S = vS[i];
                        conv_horiz = gsl_vector_alloc(U->size1);
                        h_src = (sources_svd[i])->get_map()->height;
                        h_filt = U->size1;
                        for(unsigned int r = 0 ; r < (v_tmp_svd[i])->size() ; r++)
                            {
                                dst_1D_tmp = (*(v_tmp_svd[i]))[r];
                                gsl_matrix_get_col(conv_horiz,U,r);
                                float eigen_value = gsl_vector_get(S,r); 
                                int j_min = max(int(uy - h_filt/2.0), 0);
                                int j_max = min(h_src, int(h_filt/2.0 + uy));
                                for(int j = j_min ; j < j_max ; j++)
                                    {
                                        int l = j + h_filt/2 - uy;
                                        float delta = eigen_value * du * gsl_vector_get(conv_horiz,l);
                                        float old_value = gsl_matrix_get(dst_1D_tmp,j, ux);
                                        gsl_matrix_set(dst_1D_tmp,j,ux,old_value - delta);
                                    }                                    
                            }
                        gsl_vector_free(conv_horiz);
                    }
            }
        d += du;
    }
    
    return d;
}

int 
Layer::connect(core::LayerPtr src, gsl_matrix * kernel)
{
    // Connect method used in the case of shared links

    // We store the source layer
    sources_shared.push_back(src);

    // We store the weight matrix
    kernels_shared.push_back(gsl_matrix_alloc(kernel->size1,kernel->size2));
    gsl_matrix_memcpy(kernels_shared.back(),kernel);  

    // We specifiy that this kernel is a for shared links
    types.push_back(SVD_SHARED_LINKS);

    // We return the index of the stored weights and source layer
    // This index is returned to svd::Projection
    // that carries it to the instance of svd::Link
    // It is then used in svd::Link::compute() 
    // to access to the parameters of the weigths stored in the layer
    return sources_shared.size()-1;  
}


int
Layer::connect(int rank, gsl_matrix * U, gsl_vector * S, gsl_matrix * V, core::LayerPtr src, gsl_matrix * kernel)
{
    // We define rank matrices in a vector that we push at the end of v_tmp
    
    std::vector< gsl_matrix * > * vec_tmp = new std::vector< gsl_matrix *>();
    
    for(int i = 0 ; i < rank ; i++)
        {
            vec_tmp->push_back(gsl_matrix_alloc(src->get_map()->height,src->get_map()->width));
        }
    
    v_tmp_svd.push_back(vec_tmp);

    // Matrix U
    gsl_vector * temp = gsl_vector_alloc(U->size1);
    vU.push_back(gsl_matrix_alloc(U->size1,rank));
    for(int i = 0 ; i < rank ; i++)
        {
            gsl_matrix_get_col(temp,U,i);
            gsl_matrix_set_col(vU.back(),i,temp);
        }
    
    // Vector S
    vS.push_back(gsl_vector_alloc(S->size));
    gsl_vector_memcpy(vS.back(),S);

    // Matrix V
    temp = gsl_vector_alloc(V->size1);
    vV.push_back(gsl_matrix_alloc(V->size1,rank));
    for(int i = 0 ; i < rank ; i++)
        {
            gsl_matrix_get_col(temp,V,i);
            gsl_matrix_set_col(vV.back(),i,temp);
        }

    kernels_svd.push_back(gsl_matrix_alloc(kernel->size1,kernel->size2));
    gsl_matrix_memcpy(kernels_svd.back(),kernel);
    
    sources_svd.push_back(src);
    
    // We specify that this source is for SVD links
    types.push_back(SVD_SVD_LINKS);

    return sources_svd.size()-1;
}

// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Layer::boost (void)
{
    //register_ptr_to_python< boost::shared_ptr<Layer> >();
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");  

 
    class_<Layer, bases<core::Layer> >("Layer",
                                       "======================================================================\n"
                                       "                                                                      \n"
                                       " A SVD Layer contains units that can share the weights they have with \n"
                                       " source layers.                                                       \n"
                                       " We propose two types of links : shared links and SVD links           \n"
                                       " In the two cases, the units have access to the weights of the links  \n"
                                       " that connect them to source units via their layer                    \n"
                                       " - Shared links are an optimisation for the memory size but it can    \n"
                                       " slow down the computations                                           \n"
                                       " - SVD links are optimised both for memory and execution time. It is  \n"
                                       " based on Singular Value Decomposition                                \n"
                                       " Ref : http://en.wikipedia.org/wiki/Singular_value_decomposition      \n"
                                       "======================================================================\n",
                                       init<>(
                                              "__init__() -- initializes layer\n")
                                       )
        .def("clear",&Layer::clear, "clear() : reset the activity of the units in the layer \n")
        //.def ("compute_dp", &Layer::compute_dp,
        //      "compute_dp() -> float -- computes potentials and return dp\n")
        ;
}

