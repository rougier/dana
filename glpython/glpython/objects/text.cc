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
#include "text.h"

using namespace glpython::objects;


//_________________________________________________________________________Text
Text::Text (std::string text, tuple position, tuple color, float size,
            std::string alignment, float orientation,
            float alpha, std::string name) : core::Object (name)
{
    set_text (text);
    set_color (color);
    set_position (position);
    set_size (size);
    set_alignment (alignment);
    set_orientation (orientation);
    set_alpha (alpha);
    depth = 100;
    list = 0;
    is_ortho = true;
}

//________________________________________________________________________~Text
Text::~Text (void)
{
    if (list)
        glDeleteLists (list,1);
}

//_____________________________________________________________get/set position
void
Text::set_position (tuple position)
{
    try {
        this->x = extract< float >(position[0])();
        this->y = extract< float >(position[1])();
    } catch (...) {
        PyErr_Print();
    }
}
tuple
Text::get_position (void)
{
    return make_tuple (x,y);
}

//_________________________________________________________________get/set size
void
Text::set_size (float size)
{
    this->size = fabs(size);
    
    if ((this->size > 1) and (this->size < 6))
        this->size = 6;
    else if (this->size == 0)
        this->size = 1.0f;
    this->dirty = true;    
}
float
Text::get_size (void)
{
    return size;
}

//_________________________________________________________________get/set text
void
Text::set_text (std::string text)
{   
    this->text = text;
    this->dirty = true;
}
std::string
Text::get_text (void)
{
    return text;
}

//________________________________________________________________get/set color
void
Text::set_color (tuple color)
{
    this->color = core::ColorPtr (new core::Color (color));
    this->dirty = true;    
}

tuple
Text::get_color (void)
{
    return make_tuple (color->get_red(),color->get_green(),color->get_blue());
}

//____________________________________________________________get/set alignment
void
Text::set_alignment (std::string alignment)
{
    if ((alignment == "right") || 
        (alignment == "left")  ||
        (alignment == "center")) {
        this->alignment = alignment;
    } else if ((this->alignment != "right") &&
               (this->alignment != "left")  &&
               (this->alignment != "center")) {
        this->alignment = "center";
    }
    this->dirty = true;    
}
std::string
Text::get_alignment (void)
{
    return alignment;
}

//__________________________________________________________get/set orientation
void
Text::set_orientation (float orientation)
{
    this->orientation = orientation;
    this->dirty = true;
}
float
Text::get_orientation (void)
{
    return orientation;
}

//________________________________________________________________get/set alpha
void
Text::set_alpha (float alpha)
{
    this->alpha = alpha;
    if (this->alpha > 1.0f)
        this->alpha = 1.0f;
    else if (this->alpha < 0.0f)
        this->alpha = 0.0f;
    this->dirty = true;    
}
float
Text::get_alpha (void)
{
    return alpha;
}


//_______________________________________________________________________render
void
Text::render (void)
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

    // Compute size
    int s = int(size);
    if (size <= 1.0f)
        s = int(6+sqrt(viewport[2]*viewport[3])*size*.25);

    FTFont *font = core::FontManager::get("bitstream vera sans", "texture", s);
    float llx, lly, llz, urx, ury, urz;
    font->BBox (text.c_str(), llx, lly, llz, urx, ury, urz);

    int x = 0;
    int y = 0;

    // Compute position
    if (this->x < 0)
        if (this->x <= -1)
            x = int (viewport[2] + this->x + 1);
        else
            x = int (viewport[2] + this->x*viewport[2] + 1);
    else
        if (this->x >= 1)
            x = int (this->x);
        else
            x = int (this->x*viewport[2]);
    if (this->y < 0)
        if (this->y <= -1)
            y = int (viewport[3] + this->y + 1);
        else
            y = int (viewport[3] + this->y*viewport[3] + 1);
    else
        if (this->y >= 1)
            y = int (this->y);
        else
            y = int (this->y*viewport[3]);

    if (x < 0)                x = 0;
    if (x >= (viewport[2]-1)) x = viewport[0]+viewport[2]-1;
    if (y < 0)                y = 0;
    if (y >= (viewport[3]-1)) y = viewport[1]+viewport[3]-1;

    glNewList (list,GL_COMPILE_AND_EXECUTE);
    glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
    glEnable (GL_TEXTURE_2D);
    if (font) {
        glColor4f (color->get_red(),color->get_green(),color->get_blue(),alpha);
        glPushMatrix();
        glTranslatef (x,y,0);        
        glRotatef (orientation,0,0,1);
        if (alignment == "center")
            glTranslatef (-fabs(urx-llx)/2, -fabs(ury-lly)/2, 0);
        else if (alignment == "right")
            glTranslatef (-fabs(urx-llx), -fabs(ury-lly)/2, 0);
        font->Render (text.c_str());
        glPopMatrix();
    }
    glDisable (GL_TEXTURE_2D);
    glEndList();
    dirty = false;
}

//________________________________________________________________python_export
void
Text::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Text> >();

    class_<Text, bases <core::Object> > ("Text",
    "______________________________________________________________________\n"
    "                                                                      \n"
    "A Text is used to display some text on display.                       \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "   text        - text to be displayed                                 \n"
    "   position    - text position relative to alignment                  \n"
    "   color       - text color                                           \n"
    "   size        - font size                                            \n"
    "   orientation - from 0 to 359 degrees                                \n"
    "   alignment   - 'right', 'left' or 'center'                          \n"
    "   alpha       - transparency level                                   \n"
    "   name        - name of the object                                   \n"
    "______________________________________________________________________\n",

    init < optional < std::string, tuple, tuple, float, std::string, float, float, std::string> > (
        (arg("text") = "Text",
         arg("position") = make_tuple (.5, -.1),
         arg("color") = make_tuple (0,0,0),
         arg("size")= 24.0f,
         arg("alignment") = "center",
         arg("orientation") = 0.0f,
         arg("alpha") = 1.0f,
         arg("name") = "title"),
    "__init__ (text, position, color, size, alignment, orientation, alpha, name)"))

    .add_property  ("text",         &Text::get_text,       &Text::set_text)
    .add_property  ("position",     &Text::get_position,   &Text::set_position)
    .add_property  ("color",        &Text::get_color,      &Text::set_color)
    .add_property  ("size",         &Text::get_size,       &Text::set_size)
    .add_property  ("aligment",     &Text::get_alignment,  &Text::set_alignment)
    .add_property  ("orientation",  &Text::get_orientation,&Text::set_orientation)
    .add_property  ("alpha",        &Text::get_alpha,      &Text::set_alpha)
    ;
}
