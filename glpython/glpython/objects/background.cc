//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "../core/colormaps.h"
#include "background.h"

using namespace glpython;
using namespace glpython::objects;


//___________________________________________________________________Background
Background::Background (core::ColormapPtr cmap,
                        std::string orientation,
                        float alpha, 
                        std::string name) : core::Object (name)
{
    set_cmap (cmap);
    set_alpha (alpha);
    set_orientation (orientation);
    depth = 0;
    list = 0;
}

//__________________________________________________________________~Background
Background::~Background (void)
{
    if (list)
        glDeleteLists (list,1);
}

//_________________________________________________________________get/set cmap
void
Background::set_cmap (core::ColormapPtr cmap)
{
    if (cmap->colors.size() < 2) {
        PyErr_SetString (PyExc_ValueError, 
                         "Colormap must have at least 2 values.");
        return;
    }

    this->cmap = cmap;
    this->dirty = true;   
}

core::ColormapPtr
Background::get_cmap (void)
{
    return cmap;
}

//__________________________________________________________get/set orientation
void
Background::set_orientation (std::string orientation)
{
    if (orientation == "horizontal")
        this->orientation = "horizontal";
    else
        this->orientation = "vertical";
    this->dirty = true;
}

std::string
Background::get_orientation (void)
{
    return orientation;
}

//________________________________________________________________get/set alpha
void
Background::set_alpha (float alpha)
{
    this->alpha = alpha;
    if (this->alpha > 1.0f)
        this->alpha = 1.0f;
    else if (this->alpha < 0.0f)
        this->alpha = 0.0f;
    this->dirty = true;    
}
float
Background::get_alpha (void)
{
    return alpha;
}

//_______________________________________________________________________render
void
Background::render (void)
{
    if (cmap->colors.size() < 2)
        return;

    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);
    
    if (mode == GL_SELECT)
        return;
    if (!visible)
        return;
    if (!dirty) {
        glCallList (list);
        return;
    }


    int viewport[4];
    glGetIntegerv (GL_VIEWPORT, viewport);
    int x = 0;
    int y = 0;
    int w = viewport[2];
    int h = viewport[3];

    if (list)
        glDeleteLists (list,1);
    list = glGenLists(1);

    glNewList (list, GL_COMPILE_AND_EXECUTE);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glMatrixMode (GL_PROJECTION);
    glPushMatrix ();
    glLoadIdentity ();
    glOrtho (0, w, 0, h, -1, 1);
    glMatrixMode (GL_MODELVIEW);
    glPushMatrix ();
    glLoadIdentity();
    glDisable (GL_DEPTH_TEST);

    // Colors
    for (unsigned int i=0; i<(cmap->colors.size()-1); i++) {
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

        glBegin (GL_QUADS);        
        if (orientation == "vertical") {
            x1 = x;
            x2 = x+w;
            y1 = y+int(h*cmap->colors[i].data[core::VALUE]);
            y2 = y+int(h*cmap->colors[i+1].data[core::VALUE]);
            glColor4f (cmap->colors[i].data[0], cmap->colors[i].data[1],
                       cmap->colors[i].data[2], cmap->colors[i].data[3]*alpha);
            glVertex2i (x1, y1);
            glVertex2i (x2, y1);
            glColor4f (cmap->colors[i+1].data[0], cmap->colors[i+1].data[1],
                       cmap->colors[i+1].data[2], cmap->colors[i+1].data[3]*alpha);
            glVertex2i (x2, y2);
            glVertex2i (x1, y2);
        } else {
            x1 = x+int(w*cmap->colors[i].data[core::VALUE]);
            x2 = x+int(w*cmap->colors[i+1].data[core::VALUE]);
            y1 = y;
            y2 = y+h;
            glColor4f (cmap->colors[i].data[0], cmap->colors[i].data[1],
                       cmap->colors[i].data[2], cmap->colors[i].data[3]*alpha);
            glVertex2i (x1, y1);
            glVertex2i (x1, y2);
            glColor4f (cmap->colors[i+1].data[0], cmap->colors[i+1].data[1],
                       cmap->colors[i+1].data[2], cmap->colors[i+1].data[3]*alpha);
            glVertex2i (x2, y2);
            glVertex2i (x2, y1);
        }
        glEnd();
    }

    glEnable (GL_DEPTH_TEST);
    glMatrixMode (GL_PROJECTION);
    glPopMatrix ();
    glMatrixMode (GL_MODELVIEW);
    glPopMatrix();    
    glEndList();
    dirty = false;
}

//________________________________________________________________python_export
void
Background::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Background> >();

    class_<Background, bases <core::Object> >("Background",
    " ______________________________________________________________________\n"
    "                                                                       \n"
    " ______________________________________________________________________\n",
     
    init< core::ColormapPtr, optional <std::string, float, std::string > > (
        (arg("cmap") = core::Colormaps::Default,
         arg("orientation") = "vertical",
         arg("alpha") = 1.0,
         arg("name") = "Background"),
        "__init__ (cmap, orientation, alpha, name)\n"))

    .add_property ("cmap",         &Background::get_cmap,        &Background::set_cmap)
    .add_property ("orientation",  &Background::get_orientation, &Background::set_orientation)
    .add_property ("alpha",        &Background::get_alpha,       &Background::set_alpha)
    ;       
}
