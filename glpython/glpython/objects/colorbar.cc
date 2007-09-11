//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <cmath>
#include "../core/font_manager.h"
#include "colorbar.h"

using namespace glpython;
using namespace glpython::objects;


//_____________________________________________________________________Colorbar
Colorbar::Colorbar (core::ColormapPtr cmap,
                    std::string title,
                    float size,
                    float aspect,
                    tuple position,                    
                    std::string orientation,
                    float alpha,
                    std::string name) : core::Object (name)
{
    set_cmap (cmap);
    set_title (title);
    set_position (position);
    set_size (size);
    set_aspect (aspect);
    set_orientation (orientation);
    set_alpha(alpha);
    depth = 75;
    list = 0;
    is_ortho = true;
}

//____________________________________________________________________~Colorbar
Colorbar::~Colorbar (void)
{
    if (list)
        glDeleteLists (list,1);
}
            
//_________________________________________________________________get/set cmap
void
Colorbar::set_cmap (core::ColormapPtr cmap)
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
Colorbar::get_cmap (void)
{
    return cmap;
}

//________________________________________________________________get/set title
void
Colorbar::set_title (std::string title)
{
    this->title = title;
    this->dirty = true;
}

std::string
Colorbar::get_title (void)
{
    return title;
}

//_________________________________________________________________get/set size
void
Colorbar::set_size (float size)
{
    if ((size > 0) && (size < 1))
        this->size = size;
    this->dirty = true;
}

float
Colorbar::get_size (void)
{
    return size;
}

//_______________________________________________________________get/set aspect
void
Colorbar::set_aspect (float aspect)
{
    if ((aspect > 1) && (size < 50))
        this->aspect = aspect;
    this->dirty = true;
}

float
Colorbar::get_aspect (void)
{
    return aspect;
}

//_____________________________________________________________get/set position
void
Colorbar::set_position (tuple position)
{
    try {
        this->x = extract< float >(position[0])();
        this->y = extract< float >(position[1])();
    } catch (...) {
        PyErr_Print();
    }
    this->dirty = true;
}

tuple
Colorbar::get_position (void)
{
    return make_tuple (x,y);
}

//__________________________________________________________get/set orientation
void
Colorbar::set_orientation (std::string orientation)
{
    if (orientation == "horizontal")
        this->orientation = "horizontal";
    else
        this->orientation = "vertical";
    this->dirty = true;
}

std::string
Colorbar::get_orientation (void)
{
    return orientation;
}

//________________________________________________________________get/set alpha
void
Colorbar::set_alpha (float alpha)
{
    this->alpha = alpha;
    if (this->alpha > 1.0f)
        this->alpha = 1.0f;
    else if (this->alpha < 0.0f)
        this->alpha = 0.0f;
    this->dirty = true;    
}
float
Colorbar::get_alpha (void)
{
    return alpha;
}


//_______________________________________________________________________render
void
Colorbar::render (void)
{
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

    if (list)
        glDeleteLists (list,1);
    list = glGenLists(1);

    int x,y,w,h;

    // Compute size
    if (orientation == "horizontal") {
        if (size > 1.0f)
            w = int (size);
        else
            w = int (size * viewport[2]);
        h = int (w / aspect);
    } else {
        if (size > 1.0f)
            h = int (size);
        else
            h = int (size * viewport[3]);
        w = int (h / aspect);
    }

    // Compute position
    if (this->x < 0)
        if (this->x <= -1)
            x = int (viewport[2] + this->x + 1 - w);
        else
            x = int (viewport[2] + this->x*viewport[2] + 1 - w);
    else
        if (this->x >= 1)
            x = int (this->x);
        else
            x = int (this->x*viewport[2]);

    if (this->y < 0)
        if (this->y <= -1)
            y = int (viewport[3] + this->y + 1 - h);
        else
            y = int (viewport[3] + this->y*viewport[3] + 1 - h);
    else
        if (this->y >= 1)
            y = int (this->y);
        else
            y = int (this->y*viewport[3]);

    if (x < 0)                x = 0;
    if (x >= (viewport[2]-1)) x = viewport[2]-1;
    if (y < 0)                y = 0;
    if (y >= (viewport[3]-1)) y = viewport[3]-1;

    glNewList (list,GL_COMPILE_AND_EXECUTE);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glColor4f (1,1,1,1);
    // Colors
    for (unsigned int i=0; i<(cmap->colors.size()-1); i++) {
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

        glBegin (GL_QUADS);        
        if (orientation == "vertical") {
            x1 = x;
            x2 = x+w;
            y1 = y+int(h*cmap->colors[i].data[core::VALUE]);
            y2 = y+int(h*cmap->colors[i+1].data[core::VALUE]);
            glColor4fv (cmap->colors[i].data);
            glVertex2i (x1, y1);
            glVertex2i (x2, y1);
            glColor4fv (cmap->colors[i+1].data);
            glVertex2i (x2, y2);
            glVertex2i (x1, y2);
        } else {
            x1 = x+int(w*cmap->colors[i].data[core::VALUE]);
            x2 = x+int(w*cmap->colors[i+1].data[core::VALUE]);
            y1 = y;
            y2 = y+h;
            glColor4fv (cmap->colors[i].data);
            glVertex2i (x1, y1);
            glVertex2i (x1, y2);
            glColor4fv (cmap->colors[i+1].data);
            glVertex2i (x2, y2);
            glVertex2i (x2, y1);
        }
        glEnd();
    }

    // Ticks
    glColor4f (0,0,0,1);
    for (unsigned int i=0; i<cmap->colors.size(); i++) {
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        if (orientation == "vertical") {
            x1 = x+w;
            x2 = int(x+w+.25*w);
            y1 = y+int(h*cmap->colors[i].data[core::VALUE]);
            y2 = y1;
        } else {
            x1 = x+int(w*cmap->colors[i].data[core::VALUE]);
            x2 = x1;
            y1 = y+h;
            y2 = int(y+h+.25*h);
        }        
        glBegin (GL_LINES);
        glVertex2i (x1, y1);
        glVertex2i (x2, y2);
        glEnd();
    }
    
    
    // Border
    glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
    glBegin(GL_QUADS);
    glVertex2i (x,   y);
    glVertex2i (x+w, y);
    glVertex2i (x+w, y+h);
    glVertex2i (x,   y+h);    
    glEnd();

    // Numbers
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glEnable (GL_TEXTURE_2D);
    
    int s = int(6+sqrt(viewport[2]*viewport[3])*.0125);
    
    FTFont *font = core::FontManager::get("bitstream vera sans mono", "texture", s);
    if (font) {
        glColor4f (0,0,0,1);
        char text[32];

        for (unsigned int i=0; i<cmap->colors.size(); i++) {
            float v = cmap->min +
               cmap->colors[i].data[core::VALUE]*(cmap->max-cmap->min);
            if (v >= 0)
                sprintf (text, "+%.2f", v);
            else
                sprintf (text, "-%.2f", fabs(v));
            int x1,y1;
            float lx, ly, lz, ux, uy, uz;
            font->BBox (text, lx, ly, lz, ux, uy, uz);

            if (orientation == "vertical") {
                x1 = int(x+w*1.30);
                y1 = y+int(h*cmap->colors[i].data[core::VALUE])
                     - int(fabs(uy-ly)/2.0);
            } else {
                x1 = x+int(w*cmap->colors[i].data[core::VALUE])
                   - int(fabs(ux-lx)/2.0);
                y1 = int(y+h*1.30);
            }
            glPushMatrix();
            glTranslatef (x1,y1,0);
            font->Render(text);
            glPopMatrix();
         }
    }

    font = core::FontManager::get("bitstream vera sans", "texture", s);
    if ((font) && (!title.empty())) {
        int x1, y1;
        float lx, ly, lz, ux, uy, uz;
        font->BBox (title.c_str(), lx, ly, lz, ux, uy, uz);
        if (orientation == "horizontal") {
            x1 = int (x+w/2 - fabs(ux-lx)/2.0);
            y1 = int (y-h*.25 - fabs(uy-ly));
            glPushMatrix();
            glTranslatef (x1,y1,0);
            font->Render(title.c_str());
            glPopMatrix(); 
        } else {
            x1 = int (x-w*.25 - fabs(uy-ly));
            y1 = int (y+h/2 - fabs(ux-lx)/2.0);
            glPushMatrix();
            glTranslatef (x1,y1,0);
            glRotatef (90,0,0,1);
            font->Render(title.c_str());
            glPopMatrix(); 
        }
    }

    glDisable (GL_TEXTURE_2D);
    glEndList();
    dirty = false;
}

//________________________________________________________________python_export
void
Colorbar::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Colorbar> >();

    class_<Colorbar, bases <core::Object> >("Colorbar",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A colorbar is used to represents a colormap as a rectangular area of  \n"
    "the display. Size and position of the colorscale can be set either in \n"
    "absolute or relative coordinates.                                     \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "   cmap -- colormap the colorbar must display                         \n"
    "   title -- title to be displayed                                     \n"
    "   size -- relative or absolute size                                  \n"
    "   aspect -- colorbar aspect                                          \n"
    "   position -- relative or absolute position                          \n"
    "   orientation -- either 'horizontal' or 'vertical'                   \n"
    "   alpha -- transparency of the colorbar                              \n"
    "______________________________________________________________________\n",
    init< optional <core::ColormapPtr, std::string, float, float,
                    tuple, std::string, float, std::string > > (
        (arg("cmap") = core::Colormaps::Default,
         arg("title") = "",
         arg("size") = .8,
         arg("aspect")  = 20.0,
         arg("position") = make_tuple (-0.12,.1),
         arg("orientation") = "vertical",
         arg("alpha") = 1.0,
         arg("name") = "Colorbar"),
        "__init__ (cmap, size, aspect, position, orientation, alpha, name)\n"))

    .add_property ("cmap",         &Colorbar::get_cmap,        &Colorbar::set_cmap)
    .add_property ("title",        &Colorbar::get_title,       &Colorbar::set_title)
    .add_property ("size",         &Colorbar::get_size,        &Colorbar::set_size)
    .add_property ("aspect",       &Colorbar::get_aspect,      &Colorbar::set_aspect)
    .add_property ("position",     &Colorbar::get_position,    &Colorbar::set_position)
    .add_property ("orientation",  &Colorbar::get_orientation, &Colorbar::set_orientation)
    .add_property ("alpha",        &Colorbar::get_alpha,       &Colorbar::set_alpha)
    ;
}
