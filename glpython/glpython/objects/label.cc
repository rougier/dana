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
#include "label.h"

using namespace glpython::objects;


//________________________________________________________________________Label
Label::Label (std::string text, tuple position,
              tuple fg_color, tuple bg_color, tuple br_color,
              float size, float alpha, std::string name) : core::Object (name)
{
    set_text (text);
    set_fg_color (fg_color);
    set_bg_color (bg_color);
    set_br_color (br_color);
    set_position (position);
    set_size (size);
    set_alpha (alpha);
    depth = 100;
    list = 0;
    is_ortho = false;
}

//_______________________________________________________________________~Label
Label::~Label (void)
{
    if (list)
        glDeleteLists (list,1);
}

//_____________________________________________________________get/set position
void
Label::set_position (tuple position)
{
    try {
        this->x  = extract< float >(position[0])();
        this->y  = extract< float >(position[1])();
        this->z1 = extract< float >(position[2])();
        this->z2 = extract< float >(position[3])();
    } catch (...) {
        PyErr_Print();
    }
}
tuple
Label::get_position (void)
{
    return make_tuple (x,y, z1, z2);
}

//_________________________________________________________________get/set size
void
Label::set_size (float size)
{
    this->size = fabs(size);
    
    if ((this->size > 1) and (this->size < 6))
        this->size = 6;
    else if (this->size == 0)
        this->size = 1.0f;
    this->dirty = true;    
}
float
Label::get_size (void)
{
    return size;
}

//_________________________________________________________________get/set text
void
Label::set_text (std::string text)
{   
    this->text = text;
    this->dirty = true;
}
std::string
Label::get_text (void)
{
    return text;
}

//_____________________________________________________________get/set fg_color
void
Label::set_fg_color (tuple color)
{
    this->fg_color = core::ColorPtr (new core::Color (color));
    this->dirty = true;    
}

tuple
Label::get_fg_color (void)
{
    return make_tuple (fg_color->get_red(), fg_color->get_green(),
                       fg_color->get_blue(), fg_color->get_alpha());
}

//_____________________________________________________________get/set bg_color
void
Label::set_bg_color (tuple color)
{
    this->bg_color = core::ColorPtr (new core::Color (color));
    this->dirty = true;    
}

tuple
Label::get_bg_color (void)
{
    return make_tuple (bg_color->get_red(), bg_color->get_green(),
                       bg_color->get_blue(), bg_color->get_alpha());
}

//_____________________________________________________________get/set br_color
void
Label::set_br_color (tuple color)
{
    this->br_color = core::ColorPtr (new core::Color (color));
    this->dirty = true;    
}

tuple
Label::get_br_color (void)
{
    return make_tuple (br_color->get_red(), br_color->get_green(),
                       br_color->get_blue(), br_color->get_alpha());
}

//________________________________________________________________get/set alpha
void
Label::set_alpha (float alpha)
{
    this->alpha = alpha;
    if (this->alpha > 1.0f)
        this->alpha = 1.0f;
    else if (this->alpha < 0.0f)
        this->alpha = 0.0f;
    this->dirty = true;    
}
float
Label::get_alpha (void)
{
    return alpha;
}


//_______________________________________________________________________render
void
Label::render (void)
{
    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);
    if (mode == GL_SELECT)
        return;
     
    if (!visible)
        return;

    int viewport[4];
    glGetIntegerv (GL_VIEWPORT, viewport);

    // Compute size
    int s = int(size);
    if (size <= 1.0f)
        s = int(6+sqrt(viewport[2]*viewport[3])*size*.25);

    FTFont *font = core::FontManager::get("bitstream vera sans", "texture", s);
    float llx, lly, llz, urx, ury, urz;
    font->BBox (text.c_str(), llx, lly, llz, urx, ury, urz);

    float scale = .00025*s;
    glDisable (GL_TEXTURE_RECTANGLE_ARB);
    glDisable (GL_TEXTURE_2D);
    if (font) {
    	glPushMatrix ();

        // Line
        glColor4f (br_color->get_red(),br_color->get_green(),br_color->get_blue(),alpha);
        glBegin (GL_LINES);
        glVertex3f (x, y, z1);
        glVertex3f (x, y, z2);
        glEnd();
        
        glTranslatef (x,y,z2);

        float m[16];
        glGetFloatv (GL_MODELVIEW_MATRIX, m);
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                if (i == j)
                    m[i*4+j] = 1; //m[i*4+j];
                else
                    m[i*4+j] = 0.0f;
    	glPushMatrix ();
        glLoadMatrixf (m);
    	glScalef (scale,scale,1);
        glTranslatef (-fabs(urx-llx)/2, -fabs(ury-lly)/2, 0);
        
        // Background
        glColor4f (bg_color->get_red(),bg_color->get_green(),bg_color->get_blue(),alpha);
    	glPolygonOffset (1.0f, 1.0f);
	    glEnable (GL_POLYGON_OFFSET_FILL);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glBegin (GL_QUADS);
        glVertex3f (llx-.05/scale, lly-.01/scale, 0);
        glVertex3f (urx+.05/scale, lly-.01/scale, 0);
        glVertex3f (urx+.05/scale, ury+.01/scale, 0);
        glVertex3f (llx-.05/scale, ury+.01/scale, 0);
        glEnd();

        // Border
	    glDisable (GL_POLYGON_OFFSET_FILL);
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        glColor4f (br_color->get_red(),br_color->get_green(),br_color->get_blue(),alpha);
        glBegin (GL_QUADS);
        glVertex3f (llx-.05/scale, lly-.01/scale, 0);
        glVertex3f (urx+.05/scale, lly-.01/scale, 0);
        glVertex3f (urx+.05/scale, ury+.01/scale, 0);
        glVertex3f (llx-.05/scale, ury+.01/scale, 0);
        glEnd();

        // Text
        glEnable (GL_TEXTURE_2D);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glColor4f (fg_color->get_red(),fg_color->get_green(),fg_color->get_blue(),alpha);
        font->Render (text.c_str());
        glDisable (GL_TEXTURE_2D);

        glPopMatrix();
        glPopMatrix();
    }
    dirty = false;
}

//________________________________________________________________python_export
void
Label::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Label> >();

    class_<Label, bases <core::Object> > ("Label",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A Label is used to display some text                                  \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "   text        - text to be displayed                                 \n"
    "   position    - label position as (x,y,z1,z2)                        \n"
    "   fg_color    - text color                                           \n"
    "   bg_color    - background color                                     \n"
    "   br_color    - border color                                         \n"
    "   size        - font size                                            \n"
    "   alpha       - transparency level                                   \n"
    "   name        - name of the object                                   \n"
    "______________________________________________________________________\n",

    init < optional < std::string, tuple, tuple, tuple, tuple, float, float, std::string> > (
        (arg("text") = "Label",
         arg("position") = make_tuple (0, 0, 0, .5),
         arg("fg_color") = make_tuple (0,0,0,1),
         arg("bg_color") = make_tuple (1,1,1,1),
         arg("br_color") = make_tuple (0,0,0,1),
         arg("size")= 12.0f,
         arg("alpha") = 1.0f,
         arg("name") = "title"),
    "__init__ (text, position, color, size, alignment, orientation, alpha, name)"))

    .add_property  ("text",         &Label::get_text,       &Label::set_text)
    .add_property  ("position",     &Label::get_position,   &Label::set_position)
    .add_property  ("fg_color",     &Label::get_fg_color,   &Label::set_fg_color)
    .add_property  ("bg_color",     &Label::get_bg_color,   &Label::set_bg_color)
    .add_property  ("br_color",     &Label::get_br_color,   &Label::set_br_color)
    .add_property  ("size",         &Label::get_size,       &Label::set_size)
    .add_property  ("alpha",        &Label::get_alpha,      &Label::set_alpha)
    ;
}
