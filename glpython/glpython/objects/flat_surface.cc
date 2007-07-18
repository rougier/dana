//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "flat_surface.h"

using namespace boost::python;
using namespace glpython::objects;


//__________________________________________________________________FlatSurface
FlatSurface::FlatSurface (numeric::array X,
              tuple frame,
              core::ColormapPtr colormap,
              float alpha,
              bool has_grid,
              std::string name) : Array (X,frame,colormap,alpha,has_grid,name)

{
    tex_id = 0;
}


//_________________________________________________________________~FlatSurface
FlatSurface::~FlatSurface (void)
{
    if (tex_id)
        glDeleteTextures (1, &tex_id);
}

//___________________________________________________________________initialize
void
FlatSurface::initialize (void)
{
    Array::initialize();
    if (this->tex_id)
        glDeleteTextures (1, &this->tex_id);
    
    glGenTextures (1, &this->tex_id);
    glEnable (GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture (GL_TEXTURE_RECTANGLE_ARB, this->tex_id);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                 GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                 GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri (GL_TEXTURE_RECTANGLE_ARB,
	                 GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, d0, d1,
                  0, GL_RGBA, GL_FLOAT, 0);
    glDisable (GL_TEXTURE_RECTANGLE_ARB);
    update();
}

//_______________________________________________________________________render
void
FlatSurface::render (void)
{
    if (!tex_id)
        initialize();

    if (dirty)
        update();

    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);

    int d1 = array->dimensions[0];
    int d0 = array->dimensions[1];
    float x, y, w, h;
    try {
        x = extract< float >(this->frame[0])();
        y = extract< float >(this->frame[1])();
        w = extract< float >(this->frame[2])();
        h = extract< float >(this->frame[3])();
    } catch (...) {
        PyErr_Print();
        return;
    }

    glPushAttrib (GL_ENABLE_BIT);
    glDisable (GL_LIGHTING);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);

    if (mode == GL_RENDER) {         
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable (GL_BLEND);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glEnable (GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture (GL_TEXTURE_RECTANGLE_ARB, this->tex_id);
        glPolygonOffset (1,1);
        glEnable (GL_POLYGON_OFFSET_FILL);

        // Array
        glColor4f (1,1,1,this->alpha);
        glBegin (GL_QUADS);
        glTexCoord2f(0, d1);  glVertex2f (-0.5+x,   -0.5+y);
        glTexCoord2f(d0, d1); glVertex2f (-0.5+x+w, -0.5+y);
        glTexCoord2f(d0, 0);  glVertex2f (-0.5+x+w, -0.5+y+h);
        glTexCoord2f(0, 0);   glVertex2f (-0.5+x,   -0.5+y+h);
        glEnd ();

        glDisable (GL_BLEND);
        glDisable (GL_TEXTURE_RECTANGLE_ARB);
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        glDisable (GL_POLYGON_OFFSET_FILL);

        // Selection
        if ((sx > -1) && (sy > -1)) {
            float dx = float(w)/float(d0);
            float dy = float(h)/float(d1);
            glColor3f (0,0,0);
            glBegin (GL_QUADS);
            glVertex2f (-0.5+x+sx*dx,     -0.5+y+sy*dy);
            glVertex2f (-0.5+x+(sx+1)*dx, -0.5+y+sy*dy);
            glVertex2f (-0.5+x+(sx+1)*dx, -0.5+y+(sy+1)*dy);
            glVertex2f (-0.5+x+sx*dx,     -0.5+y+(sy+1)*dy);
            glEnd ();
        }

        // Border
        glColor3f (0,0,0);
        glBegin (GL_QUADS);
        glVertex2f (-0.5+x,   -0.5+y);
        glVertex2f (-0.5+x+w, -0.5+y);
        glVertex2f (-0.5+x+w, -0.5+y+h);
        glVertex2f (-0.5+x,   -0.5+y+h);
        glEnd ();
        
    } else if (mode == GL_SELECT) {
        float dx = float(w)/float(d0);
        float dy = float(h)/float(d1);
        for (int j=0; j<d1; j++) {
            for (int i=0; i<d0; i++) {
                glLoadName (id + (d1-1-j)*d0+i);
                glBegin (GL_QUADS);
                glVertex2f (-0.5+x+i*dx,     -0.5+y+(d1-j)*dy);
                glVertex2f (-0.5+x+(i+1)*dx, -0.5+y+(d1-j)*dy);
                glVertex2f (-0.5+x+(i+1)*dx, -0.5+y+(d1-j-1)*dy);
                glVertex2f (-0.5+x+i*dx,     -0.5+y+(d1-j-1)*dy);
                glEnd ();
            }
        }
        glLoadName (0);        
    }
    glPopAttrib ();
}

//_______________________________________________________________________update
void
FlatSurface::update (void)
{
    Array::update();
    if (!tex_id)
        return;

    glEnable (GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, tex_id);
    glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, d0, d1,
                     GL_RGBA, GL_FLOAT, (GLvoid *) data);
}
//________________________________________________________________python_export
void
FlatSurface::python_export (void)
{

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<FlatSurface> >();
    import_array();
    numeric::array::set_module_and_type ("numpy", "ndarray");  

    class_<FlatSurface, bases< Array> > ("FlatSurface",
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
         arg("name") = "FlatSurface"),
        "__init__ ( X, frame, cmap, alpha, name )\n"))
    ;
}
