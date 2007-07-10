//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include "core/map.h"
#include "core/layer.h"
#include "core/link.h"
#include "cnft/spec.h"
#include "layer.h"
#include "link.h"
#include "unit.h"

// To perform the Singular Value Decomposition for the optimized convolution
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>

using namespace boost::python::numeric;
using namespace dana;
using namespace dana::svd;

// ============================================================================
//  constructor
// ============================================================================
Unit::Unit(void) : cnft::Unit()
{}

// ============================================================================
//  destructor
// ============================================================================
Unit::~Unit(void)
{}

// ============================================================================
//  Get weights from layer as a numpy::array
// ============================================================================
object
Unit::get_weights (const core::LayerPtr layer)
{
    if (layer == object()) {
        PyErr_SetString(PyExc_AssertionError, "layer is None");
        throw_error_already_set();
        return object();
    }

    if ((layer->map == 0) || (layer->map->width == 0)) {
        PyErr_SetString(PyExc_AssertionError, "layer has no shape");
        throw_error_already_set();
        return object();
    }

    // We check whether or not the source layer is contained by the destination layer,
    // the destination layer being the layer containing this unit
    // If this is the case, it means that the weights are contained by the layer,
    // they cannot be obtained by browsing the afferent or lateral vectors

    // Soucis ??
    // Quand on passe par un LayerPtr.get(), on a un "glibc : python free()" error à l'exécution
    svd::Layer * this_layer = (svd::Layer *) (this->layer);

    int type;
    int index_src = this_layer->is_source(layer,type);
    
    int width = layer->map->width;
    int height = layer->map->height; 
    
    npy_intp dims[2] = {height, width};
    object obj(handle<>(PyArray_SimpleNew (2, dims, PyArray_FLOAT)));
    PyArrayObject *array = (PyArrayObject *) obj.ptr();
    PyArray_FILLWBYTE(array, 0);
    float *data = (float *) array->data;       

    if(index_src != -1)
        {
            // If index_src is different from -1, the links are shared
            // or computed by the SVD.
            // In these two cases, the layer contains a copy of the kernel that
            // we can use to display the weights
            gsl_matrix * kernel = this_layer->get_kernel(index_src,type);
            int h_filt = kernel->size1;
            int w_filt = kernel->size2;
            int pos_x = x;
            int pos_y = y;
            for(int i = 0 ; i < height ; i++)
                {
                    for(int j = 0 ; j < width ; j++)
                        {
                            if((i+h_filt/2.0 - pos_y >=0)
                               && (i+h_filt/2.0 - pos_y < h_filt)
                               && (j+w_filt/2.0 - pos_x >= 0)
                               && (j+w_filt/2.0 - pos_x < w_filt))
                                {
                                    data[i*width + j] = gsl_matrix_get(kernel,int(i+h_filt/2.0 - pos_y),int(j+w_filt/2.0 - pos_x));
                                }
                        }
                }
        }
    else
        {
            // Otherwise, it is a core::Link
            const std::vector<core::LinkPtr> *wts;
            if (layer.get() == this->layer) {
                wts = &laterals;
            } else {
                wts = &afferents;
            }
            for (unsigned int i=0; i< wts->size(); i++) {
                core::UnitPtr unit = wts->at(i)->source;
                if (unit->layer == layer.get())
                    if ((unit->y > -1) && (unit->x > -1))
                        data[unit->y*width+unit->x] += wts->at(i)->weight;
            }
        }
    return extract<numeric::array>(obj);  
}


// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Unit::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();

    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");  
    
    class_<Unit, bases<cnft::Unit> >("Unit",
                                     "======================================================================\n"
                                     "                                                                      \n"
                                     " An svd::Unit overloads a cnft::Unit \n"
                                     " It then inherits all the methods. Only get_weights is overloaded     \n"
                                     " because of the specific way the weights are reprented. In the case of\n"
                                     " shared or svd links defined with svd::Projection, these weights are  \n"
                                     " contained by the layer and not by the unit itself                    \n"
                                     "                                                                      \n"
                                     "======================================================================\n",
                                     init<>(
                                            "__init__ () -- initialize unit\n")
                                     )
        ;
}
