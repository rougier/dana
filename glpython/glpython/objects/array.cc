//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "array.h"

using namespace boost::python;
using namespace glpython::objects;


//________________________________________________________________________Array
Array::Array (numeric::array X, tuple frame, core::ColormapPtr colormap,
              float alpha, bool has_grid, std::string name) : core::Object (name)
{
    array = 0;
    data = 0;
    this->colormap = colormap;

    if (!PyArray_Check (X.ptr())) {
        PyErr_SetString (PyExc_ValueError, "expected a PyArrayObject");
        throw_error_already_set();
        return;
    }
    
    array = (PyArrayObject *) X.ptr();
    if ((array->nd != 2) &&
       ((array->nd == 3) && ((array->dimensions[2] != 3) 
                            && (array->dimensions[2] != 4)))) {
        PyErr_SetString (PyExc_ValueError,
                         "array shape must be (M,N) (M,N,3) or (M,N,4)");
        throw_error_already_set();
        this->array = 0;
        return;
    }
    if (array->descr->type_num != NPY_FLOAT) {
        PyErr_SetString (PyExc_ValueError,
                         "array data type must be float");
        throw_error_already_set();
        array = 0;
        return;
    }

    Py_INCREF (this->array);
    this->frame = frame;
    this->alpha = alpha;
    this->has_grid = has_grid;
    sx = sy = -1;
    this->d0 = this->array->dimensions[1];
    this->d1 = this->array->dimensions[0];
}


//_______________________________________________________________________~Array
Array::~Array(void)
{
    Py_DECREF (this->array);
    delete [] data;
}

//_________________________________________________________________get/set data
void
Array::set_data (numeric::array X)
{
    if (!PyArray_Check(X.ptr())) {
        PyErr_SetString (PyExc_ValueError, "expected a PyArrayObject");
        throw_error_already_set();
        return;
    }

    PyArrayObject *ar = (PyArrayObject *) X.ptr();
    if ((ar->nd != 2) &&
           ((ar->nd == 3) && ((ar->dimensions[2] != 3) 
                          &&  (ar->dimensions[2] != 4)))) {
        PyErr_SetString(PyExc_ValueError,
                        "array shape must be (M,N) (M,N,3) or (M,N,4)");
        throw_error_already_set();
        return;
    }
    if (ar->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(PyExc_ValueError,
                        "array data type must be float");
        throw_error_already_set();
        return;
    }

    if (ar != array) {
        Py_DECREF (array);
        array = (PyArrayObject *) X.ptr();
        Py_INCREF (array);
        initialize ();
        update();
    }
}

numeric::array
Array::get_data (void)
{
    return numeric::array (this->array);
}


//________________________________________________________________get/set frame
void
Array::set_frame (tuple frame)
{
    this->frame = frame;
    initialize();
}

tuple
Array::get_frame (void)
{
    return this->frame;
}

//________________________________________________________________get/set alpha
void
Array::set_alpha (float alpha)
{
    this->alpha = alpha;
    update();
}

float
Array::get_alpha (void)
{
    return this->alpha;
}

//___________________________________________________________________initialize
void
Array::initialize (void)
{
    unsigned int d0 = this->array->dimensions[1];
    unsigned int d1 = this->array->dimensions[0];
    if ((d0 == this->d0) && (d1 == this->d1) && (this->data))
        return;

    this->d0 = d0;
    this->d1 = d1;
    id = id_counter;
    id_counter += d0*d1;
    delete [] this->data;
    this->data = new float [d0*d1*4];
    memset (this->data, 0, 4*d0*d1*sizeof(float));
}

//_______________________________________________________________________render
void
Array::render (void)
{}

//_______________________________________________________________________select
void
Array::select (int selection)
{
    if ((selection >= int(id)) && (selection < int(id+d0*d1))) {
        sx = (selection-id)%d0;
        sy = (selection-id)/d0;
        if (select_callback != object())
            select_callback (sx,sy, select_data);
    } else {
        sx = -1;
        sy = -1;
    }
}

//_______________________________________________________________________update
void
Array::update (void)
{
    if ((!array) || (!data))
        return;

    // Array shape is (M,N)
    if (array->nd == 2) {
        int s = 4*sizeof(float);
        int jj = 0;
        float v;
        for (unsigned int j=0; j<this->d1; j++) {
            int ii = 0;
            for (unsigned int i=0; i<this->d0; i++) {
                v = *(float *)(this->array->data + jj + ii);
                memcpy ((float *)&data[(this->d1-j-1)*this->d0*4+4*i],
                        this->colormap->color(v).data, s);
                ii += this->array->strides[1];
            }
            jj += this->array->strides[0];
        }
    // Array shape is (M,N,3)
    } else if ((array->nd == 3) && (array->dimensions[2] == 3)) {
        int s = 3*sizeof(float);
        int jj = 0;
        float *v;
        for (unsigned int j=0; j<this->d1; j++) {
            int ii = 0;
            for (unsigned int i=0; i<this->d0; i++) {
                v = (float *)(this->array->data + jj + ii);
                memcpy ((float *) &data[(d1-j-1)*d0*4+4*i], v, s);
                data[(d1-j-1)*d0*4+4*i+3] = alpha;
                ii += this->array->strides[1];
            }
            jj += this->array->strides[0];
        }
    // Array shape is (M,N,4)
    } else if ((array->nd == 3) && (array->dimensions[2] == 4)) {
        int s = 4*sizeof(float);
        int jj = 0;
        float *v;
        for (unsigned int j=0; j<this->d1; j++) {
            int ii = 0;
            for (unsigned int i=0; i<this->d0; i++) {
                v = (float *)(this->array->data + jj + ii);
                memcpy ((float *) &data[(d1-j-1)*d0*4+4*i], v, s);
                data[(d1-j-1)*d0*4+4*i+3] *= alpha;
                ii += this->array->strides[1];
            }
            jj += this->array->strides[0];
        }
    }
}

//______________________________________________________________________connect
void
Array::connect (std::string event, object callback, object data)
{
    if (event == "select_event") {
        select_callback = callback;
        select_data = data;
    }
}
//________________________________________________________________python_export
void
Array::python_export (void)
{

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Array> >();
    import_array();
    numeric::array::set_module_and_type ("numpy", "ndarray");  

    class_<Array, bases< core::Object> > ("Array",
    " ______________________________________________________________________\n"
    "                                                                       \n"
    " ______________________________________________________________________\n",
    init<numeric::array,
         optional <tuple, core::ColormapPtr, float, bool, std::string> > (
        (arg("X"),
         arg("frame") = make_tuple (0,0,1,1),
         arg("cmap")  = core::Colormaps::Default,
         arg("alpha") = 1,
         arg("has_grid") = true,
         arg("name") = "Array"),
        "__init__ ( X, frame, cmap, alpha, has_grid, name )\n"))

    .add_property  ("data",     &Array::get_data,  &Array::set_data)
    .add_property  ("frame",    &Array::get_frame, &Array::set_frame)
    .add_property  ("alpha",    &Array::get_alpha, &Array::set_alpha)
    .def_readwrite ("has_grid", &Array::has_grid)
    .def_readwrite ("colormap", &Array::colormap)
    .def           ("connect",  &Array::connect)
    ;
}
