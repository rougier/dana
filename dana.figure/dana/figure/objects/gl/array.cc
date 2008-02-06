// -----------------------------------------------------------------------------
// DANA 
// Copyright (C) 2006-2007  Nicolas P. Rougier
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// this program.  If not, see <http://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------

#include <GL/gl.h>
#include "array.h"
#include <numpy/arrayobject.h>

using namespace dana::figure;


Array::Array (numpy::array X, ColormapPtr cmap, py::tuple position, py::tuple size) : Object ()
{
    set_position (position);
    set_size (size);
    set_visible (true);
    set_name ("Array");
    array_ = 0;
    data_ = 0;
    cmap_ = cmap;
    set_data (X);
    tex_id_ = 0;
}


Array::~Array(void)
{
    if (array_)
        Py_DECREF (array_);
    delete [] data_;
    if (tex_id_)
        glDeleteTextures (1, &tex_id_);

}

void
Array::set_data (numpy::array X)
{
    if ((PyArrayObject *) X.ptr() != array_) {
        if (array_) {
            Py_DECREF (array_);
            array_ = 0;
        }

        if (!PyArray_Check (X.ptr())) {
            PyErr_SetString (PyExc_ValueError, "expected a PyArrayObject");
            py::throw_error_already_set();
            return;
        }
        array_ = (PyArrayObject *) X.ptr();
        if ((array_->nd != 2) &&
            ((array_->nd == 3) && ((array_->dimensions[2] != 3) 
                                  && (array_->dimensions[2] != 4)))) {
            PyErr_SetString (PyExc_ValueError,
                             "array shape must be (M,N) (M,N,3) or (M,N,4)");
            py::throw_error_already_set();
            array_ = 0;
            return;
        }
        if (array_->descr->type_num != NPY_FLOAT) {
            PyErr_SetString (PyExc_ValueError,
                             "array data type must be float");
            py::throw_error_already_set();
            array_ = 0;
            return;
        }
        Py_INCREF (array_);
        shape_ = Shape (array_->dimensions[1], array_->dimensions[0]);
        if (tex_id_) {
            glDeleteTextures (1, &tex_id_);
            tex_id_ = 0;
        }
    }

    int d0 = ((PyArrayObject *) X.ptr())->dimensions[1];
    int d1 = ((PyArrayObject *) X.ptr())->dimensions[01];

    if ((!data_) || (d0 != int(shape_.x)) || (d1 != int(shape_.y))) {
        d0 = array_->dimensions[1]; //int(shape_.x);
        d1 = array_->dimensions[0]; //int(shape_.y);
        data_ = new float [d0*d1*4];
        memset (data_, 0, 4*d0*d1*sizeof(float));
        id_ = id_counter_;
        id_counter_ += d0*d1;
    }
}

numpy::array
Array::get_data (void)
{
    return numpy::array (array_);
}


void
Array::update (void)
{
    if ((!array_) || (!data_))
        return;

    unsigned int d0 = array_->dimensions[1];
    unsigned int d1 = array_->dimensions[0];

    // Array shape is (M,N)
    if (array_->nd == 2) {
        int s = 4*sizeof(float);
        int jj = 0;
        float v;
        for (unsigned int j=0; j<d1; j++) {
            int ii = 0;
            for (unsigned int i=0; i<d0; i++) {
                v = *(float *)(array_->data + jj + ii);
                memcpy ((float *)&data_[(d1-j-1)*d0*4+4*i], cmap_->color(v).data, s);
                ii += array_->strides[1];
             }
             jj += array_->strides[0];
        }
    // Array shape is (M,N,3)
    } else if ((array_->nd == 3) && (array_->dimensions[2] == 3)) {
        int s = 3*sizeof(float);
        int jj = 0;
        float *v;
        for (unsigned int j=0; j<d1; j++) {
            int ii = 0;
            for (unsigned int i=0; i<d0; i++) {
                v = (float *)(array_->data + jj + ii);
                memcpy ((float *) &data_[(d1-j-1)*d0*4+4*i], v, s);
                data_[(d1-j-1)*d0*4+4*i+3] = 1.0f;
                ii += array_->strides[1];
            }
            jj += array_->strides[0];
        }
    // Array shape is (M,N,4)
    } else if ((array_->nd == 3) && (array_->dimensions[2] == 4)) {
        int s = 4*sizeof(float);
        int jj = 0;
        float *v;
        for (unsigned int j=0; j<d1; j++) {
            int ii = 0;
            for (unsigned int i=0; i<d0; i++) {
                v = (float *)(array_->data + jj + ii);
                memcpy ((float *) &data_[(d1-j-1)*d0*4+4*i], v, s);
                data_[(d1-j-1)*d0*4+4*i+3] = 1.0f;
                ii += array_->strides[1];
            }
            jj += array_->strides[0];
        }
    }

    if (tex_id_) {
        int d0 = array_->dimensions[1]; //int(shape_.x);
        int d1 = array_->dimensions[0]; //int(shape_.y);

        glEnable (GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture (GL_TEXTURE_RECTANGLE_ARB, tex_id_);
        glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, d0, d1,
                         GL_RGBA, GL_FLOAT, (GLvoid *) data_);
        glDisable (GL_TEXTURE_RECTANGLE_ARB);
    }
}

void
Array::render (void)
{
    int d0 = array_->dimensions[1]; //int(shape_.x);
    int d1 = array_->dimensions[0]; //int(shape_.y);

    if (!tex_id_) {
        //glDeleteTextures (1, &tex_id_);

        glGenTextures (1, &tex_id_);
        glEnable (GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture (GL_TEXTURE_RECTANGLE_ARB, tex_id_);
        glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
                         GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
                         GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
                         GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
                         GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA,
                      d0, d1, 0, GL_RGBA, GL_FLOAT, 0);
        glDisable (GL_TEXTURE_RECTANGLE_ARB);
    }

    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);

    float x = position_.x;
    float y = position_.y;
    float w = size_.x;
    float h = size_.y;

    if (mode == GL_RENDER) {
        update();
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
        glEnable (GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture (GL_TEXTURE_RECTANGLE_ARB, tex_id_);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glColor3f (1,1,1);
        glBegin (GL_QUADS);
        glTexCoord2f(0, d1);  glVertex3f (x,   y,0);
        glTexCoord2f(d0, d1); glVertex3f (x+w, y,0);
        glTexCoord2f(d0, 0);  glVertex3f (x+w, y+h,0);
        glTexCoord2f(0, 0);   glVertex3f (x,   y+h,0);
        glEnd ();
        glDisable (GL_TEXTURE_RECTANGLE_ARB);

        // Border
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        glColor3f (0,0,0);
        glLineWidth (1.0f);
        glBegin (GL_QUADS);
        glVertex3f (x,   y,25);
        glVertex3f (x+w, y,25);
        glVertex3f (x+w, y+h,25);
        glVertex3f (x,   y+h,25);
        glEnd ();
        
    } else if (mode == GL_SELECT) {
        float dx = float(w)/float(d0);
        float dy = float(h)/float(d1);

        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        for (int j=0; j<d1; j++) {
            for (int i=0; i<d0; i++) {
                glLoadName (int (id_ + (d1-1-j)*d0+i));
                glBegin (GL_QUADS);
                glVertex2f (x+i*dx,     y+(d1-j)*dy);
                glVertex2f (x+(i+1)*dx, y+(d1-j)*dy);
                glVertex2f (x+(i+1)*dx, y+(d1-j-1)*dy);
                glVertex2f (x+i*dx,     y+(d1-j-1)*dy);
                glEnd ();
            }
        }
        glLoadName (0);        
    }
}



void
Array::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Array> >();
    import_array();
    numpy::array::set_module_and_type ("numpy", "ndarray");  

    class_<Array, bases <Object> >
        ("Array",
         init < numpy::array, ColormapPtr,  tuple, tuple >
         ("__init__ (X, cmap, position, size)"))

        .add_property ("data", &Array::get_data, &Array::set_data)
         ;
}
