//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: array.cc 160 2007-05-11 12:50:33Z rougier $

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <GL/gl.h>
#include "array.h"

using namespace boost::python::numeric;
using namespace dana::gl;


// ============================================================================
//  Static variables
// ============================================================================
int Array::id_counter = 100;


// ============================================================================
//  constructor
// ============================================================================
Array::Array (object array, object frame, std::string name, int fontsize)
{
    this->array = (PyArrayObject *) array.ptr();
    Py_INCREF(this->array);
    this->frame = frame;
    this->name = name;
    this->fontsize = fontsize;
    active = true;
    visible = true;
    tex = 0;
    data = 0;
    id = id_counter++;

    cmap.add ( -1.0f, make_tuple (0.0f, 0.0f, 1.0f));
    cmap.add ( -0.5f, make_tuple (0.5f, 0.5f, 1.0f));
    cmap.add (  0.0f, make_tuple (1.0f, 1.0f, 1.0f));
    cmap.add (  0.5f, make_tuple (1.0f, 1.0f, 0.0f));
    cmap.add (  1.0f, make_tuple (1.0f, 0.0f, 0.0f));

    bool load_error = false;
    bool size_ok = true;
        
    font = new FTGLTextureFont ("/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf");
    load_error = font->Error();
    size_ok = font->FaceSize (100);
    font->CharMap (ft_encoding_unicode);
}


// ============================================================================
//  destructor
// ============================================================================
Array::~Array(void)
{
    Py_DECREF(array);
}

void
Array::set_data (object array) {
    Py_DECREF(this->array);
    this->array = (PyArrayObject *) array.ptr();
    Py_INCREF(this->array);
}

// ============================================================================
//  Opengl initialization
// ============================================================================
void
Array::initialize (void)
{
    int h  = array->dimensions[0];
    int w = array->dimensions[1];

    data = new float [w*h*4];
    memset (data, 0, 4*w*h*sizeof(float));
    glGenTextures(1, &tex);

    glEnable (GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture (GL_TEXTURE_RECTANGLE_ARB, tex);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                 GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                 GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                 GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, w, h,
                  0, GL_RGBA, GL_FLOAT, 0);
    glDisable (GL_TEXTURE_RECTANGLE_ARB);
}

// ============================================================================
//  Rendering
// ============================================================================
void
Array::render (void)
{
    
    if (data == 0)
        initialize();

    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);

    int d1 = array->dimensions[0];
    int d0 = array->dimensions[1];

    float x, y, w, h;
    try {
        x = extract< float >(frame[0])();
        y = extract< float >(frame[1])();
        w = extract< float >(frame[2])();
        h = extract< float >(frame[3])();
    } catch (...) {
        PyErr_Print();
        return;
    }

    glPushAttrib (GL_ENABLE_BIT);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);     
    glDisable (GL_LIGHTING);

    int s = 4*sizeof(float);
    if (mode == GL_RENDER) {
        int jj = 0;
        for (int j=0; j<d1; j++) {
            int ii = 0;
            for (int i=0; i<d0; i++) {
                float v = *(float *)(array->data + jj + ii);
                memcpy ((float *) &data[(d1-j-1)*d0*4+4*i], cmap.color(v).data, s);
                ii += array->strides[1];
            }
            jj += array->strides[0];
        }
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable (GL_BLEND);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glEnable (GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture (GL_TEXTURE_RECTANGLE_ARB, tex);
        glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, d0, d1,
                         GL_RGBA, GL_FLOAT, (GLvoid *) data);
        glPolygonOffset (1,1);
        glEnable (GL_POLYGON_OFFSET_FILL);
        
        // Array
        glColor4f (1,1,1,1);
        glBegin (GL_QUADS);
        glTexCoord2f(0, 0);   glVertex2f (-0.5+x,   -0.5+y);
        glTexCoord2f(d0, 0);  glVertex2f (-0.5+x+w, -0.5+y);
        glTexCoord2f(d0, d1); glVertex2f (-0.5+x+w, -0.5+y+h);
        glTexCoord2f(0, d1);  glVertex2f (-0.5+x,   -0.5+y+h);
        glEnd ();
        glDisable (GL_TEXTURE_RECTANGLE_ARB);

        // Name
        if (name != "") {
            glColor4f (0,0,0,1);
            float llx, lly, llz, urx, ury, urz;
            glEnable (GL_TEXTURE_2D);
            font->BBox (name.c_str(), llx, lly, llz, urx, ury, urz);
            GLfloat scale = .000025*fontsize;
            glPushMatrix();
            glTranslatef (-0.5+x + w/2 - (urx-llx)*scale*.5,
                          -0.5+y + h/2 - (ury-lly)*scale*.5,
                          .01);
	    	glScalef (scale, scale, 1);
	    	font->Render(name.c_str());
            glPopMatrix ();
        }

        // Border
        glDisable (GL_BLEND);
        glDisable (GL_POLYGON_OFFSET_FILL);
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        glColor3f (0,0,0);
        glBegin (GL_QUADS);
        glVertex2f (-0.5+x,   -0.5+y);
        glVertex2f (-0.5+x+w, -0.5+y);
        glVertex2f (-0.5+x+w, -0.5+y+h);
        glVertex2f (-0.5+x,   -0.5+y+h);
        glEnd ();

    } else if (mode == GL_SELECT) {
        glLoadName (id);  
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);

        glBegin (GL_QUADS);
        glVertex2f (-0.5+x,   -0.5+y);
        glVertex2f (-0.5+x+w, -0.5+y);
        glVertex2f (-0.5+x+w, -0.5+y+h);
        glVertex2f (-0.5+x,   -0.5+y+h);
        glEnd ();

        float dx = float(w)/float(d0);
        float dy = float(h)/float(d1);
        for (int j=0; j<d1; j++) {
            for (int i=0; i<d0; i++) {
                glLoadName (j*d0+i);
                glBegin (GL_QUADS);
                glVertex2f (-0.5+x+i*dx,     -0.5+y+(d1-j)*dy);
                glVertex2f (-0.5+x+(i+1)*dx, -0.5+y+(d1-j)*dy);
                glVertex2f (-0.5+x+(i+1)*dx, -0.5+y+(d1-j-1)*dy);
                glVertex2f (-0.5+x+i*dx,     -0.5+y+(d1-j-1)*dy);
                glEnd ();
            }
        }
    }
    glPopAttrib ();
}

// ============================================================================
//  
// ============================================================================
void
Array::select (int primary, int secondary)
{
    if (primary != id)
        return;
//    int d1  = array->dimensions[0];
    int d0  = array->dimensions[1];
    if (select_callback != object()) {
        select_callback (secondary%d0, secondary/d0, select_data);
    }
}


// ============================================================================
//  
// ============================================================================
void
Array::unselect (void)
{
    if (unselect_callback != object()) {
        unselect_callback (unselect_data);
    }
}

// ============================================================================
//  
// ============================================================================
void
Array::connect (std::string event, object callback, object data)
{
    if (event == "select_event") {
        select_callback = callback;
        select_data = data;

    }
    else if (event == "unselect_event") {
        unselect_callback = callback;
        unselect_data = data;
    }
}

// ============================================================================
//   boost wrapping code
// ============================================================================
void
Array::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Array> >();
    import_array();
    numeric::array::set_module_and_type ("numpy", "ndarray");  


    class_<Array> ("Array",
    "======================================================================\n"
    "\n"
    "======================================================================\n",
        init<object, optional <object, std::string, int> >(
        "__init__ () -- initialize array\n")
        )

        .def ("init", &Array::initialize)        
        .def ("render", &Array::render)
        .def ("select", &Array::select)
        .def ("unselect", &Array::unselect)
        .def ("connect", &Array::connect)
        .def ("set_data", &Array::set_data)
        .def_readwrite ("active", &Array::active)
        .def_readwrite ("visible", &Array::visible)
        .def_readwrite ("frame", &Array::frame)
        ;
}
