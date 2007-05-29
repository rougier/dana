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
#include "array_bar.h"

using namespace boost::python::numeric;
using namespace dana::gl;


// ============================================================================
//  Static variables
// ============================================================================
int ArrayBar::id_counter = 100;


// ============================================================================
//  constructor
// ============================================================================
ArrayBar::ArrayBar (object array, object frame, std::string name, int fontsize)
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

    surface_cmap.add ( make_tuple (0.0f, 0.0f, 1.0f), -1.0f);
    surface_cmap.add ( make_tuple (0.5f, 0.5f, 1.0f), -0.5f);        
    surface_cmap.add ( make_tuple (1.0f, 1.0f, 1.0f),  0.0f);
    surface_cmap.add ( make_tuple (1.0f, 1.0f, 0.0f),  0.5f);
    surface_cmap.add ( make_tuple (1.0f, 0.0f, 0.0f),  1.0f);

    line_cmap.add ( make_tuple (0.25f, 0.25f, 0.25f), -1.0f);
    line_cmap.add ( make_tuple (0.75f, 0.75f, 0.75f),  0.0f);
    line_cmap.add ( make_tuple (0.25f, 0.25f, 0.25f),  1.0f);


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
ArrayBar::~ArrayBar(void)
{
    Py_DECREF(array);
}

void
ArrayBar::set_data (object array) {
    Py_DECREF(this->array);
    this->array = (PyArrayObject *) array.ptr();
    Py_INCREF(this->array);
}

// ============================================================================
//  Opengl initialization
// ============================================================================
void
ArrayBar::initialize (void)
{
    int h  = array->dimensions[0];
    int w = array->dimensions[1];
    data = new float [w*h*4];
    memset (data, 0, 4*w*h*sizeof(float));
}

// ============================================================================
//  Rendering
// ============================================================================
void
ArrayBar::render (void)
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
    glEnable (GL_LIGHTING);
    glPolygonOffset (1,1);
    glEnable (GL_POLYGON_OFFSET_FILL);

    if (mode == GL_RENDER) {
        int jj = 0;
        float dx = float(w)/float(d0);
        float dy = float(h)/float(d1);
        float DZ = dx*10;
        
        for (int j=0; j<d1; j++) {
            int ii = 0;
            for (int i=0; i<d0; i++) {
                float dz = *(float *)(array->data + jj + ii);
                glColor4fv (surface_cmap.colorfv (dz));
                glMaterialfv (GL_FRONT, GL_AMBIENT_AND_DIFFUSE, surface_cmap.colorfv (dz));
                if (dz > 0)
                    Cube (-0.5+x+i*dx, -0.5+y+(d1-1-j)*dy, 0, dx,dy,dz*DZ);
                else
                    Cube (-0.5+x+i*dx, -0.5+y+(d1-1-j)*dy, dz*DZ, dx,dy, -dz*DZ);                
                ii += array->strides[1];
            }
            jj += array->strides[0];
        }        

        glDisable (GL_LIGHTING);
        glDisable (GL_POLYGON_OFFSET_FILL);
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE); 
        jj = 0;
        for (int j=0; j<d1; j++) {
            int ii = 0;
            for (int i=0; i<d0; i++) {
                float dz = *(float *)(array->data + jj + ii);
                glColor4fv (line_cmap.colorfv (dz));
                Cube (-0.5+x+i*dx, -0.5+y+(d1-1-j)*dy,0, dx,dy,dz*DZ);
                ii += array->strides[1];
            }
            jj += array->strides[0];
        }     

        // Name
        if (name != "") {
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL); 
            glColor4f (0,0,0,1);
            float llx, lly, llz, urx, ury, urz;
            glEnable (GL_TEXTURE_2D);
            font->BBox (name.c_str(), llx, lly, llz, urx, ury, urz);
            GLfloat scale = .000025*fontsize;
            glPushMatrix();
            glTranslatef (-0.5+x + w/2 - (urx-llx)*scale*.5,
                          -0.5+y + h/2 - (ury-lly)*scale*.5,
                          DZ*1.1);
	    	glScalef (scale, scale, 1);
	    	font->Render(name.c_str());
            glPopMatrix ();
        }
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

void
ArrayBar::Cube (float x, float y, float z, float dx, float dy, float dz)
{
    glBegin (GL_QUADS);

    // Top Face (z)
    glNormal3f ( 0.0f,  0.0f,  1.0f);
    glVertex3f (    x,     y,  z+dz);
    glVertex3f ( x+dx,     y,  z+dz);
    glVertex3f ( x+dx,  y+dy,  z+dz);
    glVertex3f (    x,  y+dy,  z+dz);

    // Bottom Face (-z)
    glNormal3f ( 0.0f,  0.0f, -1.0f);
    glVertex3f (    x,     y,     z);
    glVertex3f (    x,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,     z);
    glVertex3f ( x+dx,     y,     z);

    // Front face (x)
    glNormal3f ( 1.0f,  0.0f,  0.0f);
    glVertex3f ( x+dx,     y,     z);
    glVertex3f ( x+dx,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,  z+dz);
    glVertex3f ( x+dx,     y,  z+dz);

    // Rear Face (-x)
    glNormal3f (-1.0f,  0.0f,  0.0f);
    glVertex3f (    x,     y,     z);
    glVertex3f (    x,     y,  z+dz);
    glVertex3f (    x,  y+dy,  z+dz);
    glVertex3f (    x,  y+dy,     z);

    // Left Face (y)
    glNormal3f ( 0.0f,  1.0f,  0.0f);
    glVertex3f (    x,     y,     z);
    glVertex3f (    x,     y,  z+dz);
    glVertex3f ( x+dx,     y,  z+dz);
    glVertex3f ( x+dx,     y,     z);

    // Right Face (-y)
    glNormal3f ( 0.0f, -1.0f,  0.0f);
    glVertex3f (    x,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,     z);
    glVertex3f ( x+dx,  y+dy,  z+dz);
    glVertex3f (    x,  y+dy,  z+dz);
    glEnd();
}


// ============================================================================
//  
// ============================================================================
void
ArrayBar::select (int primary, int secondary)
{
    if (primary != id)
        return;
    int d0  = array->dimensions[1];
    if (select_callback != object()) {
        select_callback (secondary%d0, secondary/d0, select_data);
    }
}


// ============================================================================
//  
// ============================================================================
void
ArrayBar::unselect (void)
{
    if (unselect_callback != object()) {
        unselect_callback (unselect_data);
    }
}

// ============================================================================
//  
// ============================================================================
void
ArrayBar::connect (std::string event, object callback, object data)
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
ArrayBar::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<ArrayBar> >();
    import_array();
    numeric::array::set_module_and_type ("numpy", "ndarray");  


    class_<ArrayBar> ("ArrayBar",
    "======================================================================\n"
    "\n"
    "======================================================================\n",
        init<object, optional <object, std::string, int> >(
        "__init__ () -- initialize array\n")
        )

        .def ("init", &ArrayBar::initialize)        
        .def ("render", &ArrayBar::render)
        .def ("select", &ArrayBar::select)
        .def ("unselect", &ArrayBar::unselect)
        .def ("connect", &ArrayBar::connect)
        .def ("set_data", &ArrayBar::set_data)
        .def_readwrite ("active", &ArrayBar::active)
        .def_readwrite ("visible", &ArrayBar::visible)
        .def_readwrite ("frame", &ArrayBar::frame)
        ;
}
