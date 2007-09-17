//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <GL/glu.h>
#include <iostream>
#include "viewport.h"

using namespace glpython::core;


//_____________________________________________________________________Viewport
Viewport::Viewport (tuple size, tuple position, bool has_border,
                    bool is_ortho, std::string name) : Object (name)
{
    observer = ObserverPtr (new Observer());
    set_is_ortho (is_ortho);
    has_focus = false;
    child_has_focus = false;
    this->has_border = has_border;
    try {
        x = extract< float >(position[0])();
        y = extract< float >(position[1])();
        w = extract< float >(size[0])();
        h = extract< float >(size[1])();
    } catch (...) {
        PyErr_Print();
    }
    geometry[0] = 0;
    geometry[1] = 0;
    geometry[2] = 0;
    geometry[3] = 0;
}

//____________________________________________________________________~Viewport
Viewport::~Viewport (void)
{
}

//_________________________________________________________________________repr
std::string
Viewport::repr (void)
{
    std::ostringstream ost;
    ost << "[";
    for (unsigned int i=0; i<viewports.size(); i++)
        ost << viewports[i]->repr() << ", ";
    for (unsigned int i=0; i<objects.size(); i++)
        ost << objects[i]->repr() << ", ";
    ost << "]";
    return ost.str();
}

//_______________________________________________________________________append
void
Viewport::append (ViewportPtr v)
{
    viewports.push_back (ViewportPtr(v));
}

//_______________________________________________________________________append
void
Viewport::append (ObjectPtr o)
{
    std::vector<ObjectPtr>::iterator i;
    for (i=objects.begin(); i != objects.end(); i++) {
        if (o->depth < (*i)->depth) {
            objects.insert (i, ObjectPtr(o));
            return;
        }
    }
    objects.push_back (ObjectPtr(o));
}

//__________________________________________________________________________len
int
Viewport::len (void)
{
    return viewports.size() + objects.size();
}

//______________________________________________________________________getitem
ObjectPtr
Viewport::getitem (int index)
{
    int i = index;
    if (i < 0)
        i += viewports.size() + objects.size();

    if (i < int(viewports.size())) {
        return viewports[i];
    } else {
        i -= viewports.size();
        try {
            return objects[i];
        } catch (...) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            throw_error_already_set();
        }
    }

    return ObjectPtr();
}

//______________________________________________________________________delitem
void
Viewport::delitem (int index)
{
    int i = index;
    if (i < 0)
        i += viewports.size() + objects.size();

    if (i < int(viewports.size())) {
        std::vector<ViewportPtr>::iterator it = viewports.begin();
        it += i;
        viewports.erase (it);
    } else {
        i -= viewports.size();
        std::vector<ObjectPtr>::iterator it = objects.begin();
        it += i;
        try {
            objects.erase (it);
        } catch (...) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            throw_error_already_set();
        }
    }
}

//________________________________________________________________________clear
void
Viewport::clear (void)
{
    viewports.clear();
    objects.clear();
}

//___________________________________________________________________initialize
void
Viewport::initialize (void)
{
    for (unsigned int i=0; i<objects.size(); i++) {
        object o(objects[i]);
        o.attr("init")();
    }
    for (unsigned int i=0; i<viewports.size(); i++) {
        object o(viewports[i]);
        o.attr("init")();
    }
}

//_______________________________________________________________________render
void
Viewport::render (void)
{
    if (!visible)
        return;

    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);

    glPushAttrib (GL_ENABLE_BIT | GL_VIEWPORT_BIT);
    glViewport (geometry[0], geometry[1], geometry[2], geometry[3]);
    glEnable (GL_SCISSOR_TEST);
    if (!has_border)
        glScissor (geometry[0], geometry[1], geometry[2], geometry[3]);
    else
        glScissor (geometry[0]+1, geometry[1]+1, geometry[2]-1, geometry[3]-1);    
    glClear (GL_DEPTH_BUFFER_BIT);
    glEnable (GL_DEPTH_TEST);

        
    // Regular objects
    observer->push();
    for (unsigned int i=0; i<objects.size(); i++) {
        if (!objects[i]->is_ortho) {
            object o(objects[i]);
            o.attr("render")();
        }
    }
    observer->pop();

    // "Ortho" objects
    glMatrixMode (GL_PROJECTION);
    glPushMatrix ();
    glLoadIdentity ();
    glOrtho (0, geometry[2], 0, geometry[3], -1, 1);
    glMatrixMode (GL_MODELVIEW);
    glPushMatrix ();
    glLoadIdentity();
    glDisable (GL_LIGHTING);
    glDisable (GL_TEXTURE_RECTANGLE_ARB);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable (GL_DEPTH_TEST);
    for (unsigned int i=0; i<objects.size(); i++) {
        if (objects[i]->is_ortho) {
            object o(objects[i]);
            o.attr("render")();
        }
    }
    glMatrixMode (GL_PROJECTION);
    glPopMatrix ();
    glMatrixMode (GL_MODELVIEW);
    glPopMatrix();

    glEnable (GL_DEPTH_TEST);
    
    // Viewports
    for (unsigned int i=0; i<viewports.size(); i++) {
        object o(viewports[i]);
        o.attr("render")();
    }

    glDisable (GL_SCISSOR_TEST);

    if ((!has_border) || (mode == GL_SELECT)) {
        glPopAttrib();
        return;
    }
    
    glMatrixMode (GL_PROJECTION);
    glPushMatrix ();
    glLoadIdentity ();
    glOrtho (0, geometry[2], 0, geometry[3], -1, 1);
    glMatrixMode (GL_MODELVIEW);
    glPushMatrix ();
    glLoadIdentity();

    glDisable (GL_LIGHTING);
    glDisable (GL_DEPTH_TEST);
    /*
    if ((has_focus) && (!child_has_focus)) {
        glEnable (GL_LINE_STIPPLE);
        glLineStipple (1, 0xF0F0);
    } else {
        glDisable (GL_LINE_STIPPLE);
    }
    */
    glColor4f (0,0,0,1);
    glBegin (GL_LINE_LOOP);
    glVertex2i (0, 0);
    glVertex2i (geometry[2]-1, 0);
    glVertex2i (geometry[2]-1, geometry[3]-1);
    glVertex2i (0, geometry[3]-1);        
    glEnd();
    // HACK: Missing pixel on upper right corner for some unknown reason
    if (!has_focus) {
        glBegin (GL_LINES);
        glVertex2i (geometry[2]-1, geometry[3]-1);
        glVertex2i (0, geometry[3]-1);
        glEnd();
    }

    glMatrixMode (GL_PROJECTION);
    glPopMatrix ();
    glMatrixMode (GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}

//_______________________________________________________________________select
void
Viewport::select (int selection)
{
    for (unsigned int i=0; i<objects.size(); i++) {
        object o(objects[i]);
        o.attr("select")(selection);
    }
    for (unsigned int i=0; i<viewports.size(); i++) {
        object o(viewports[i]);
        o.attr("select")(selection);
    }
}

//_______________________________________________________________________update
void
Viewport::update (void)
{
    for (unsigned int i=0; i<objects.size(); i++) {
        object o(objects[i]);
        o.attr("update")();
    }
    for (unsigned int i=0; i<viewports.size(); i++) {
        object o(viewports[i]);
        o.attr("update")();
    }
}

//_________________________________________________________________set_is_ortho
void
Viewport::set_is_ortho (bool is_ortho)
{
    observer->camera->is_ortho = is_ortho;
}

//_________________________________________________________________get_is_ortho
bool
Viewport::get_is_ortho (void)
{
    return observer->camera->is_ortho;
}


//_________________________________________________________________set_position
void
Viewport::set_position (object position)
{
    try {
        this->x = extract< float >(position[0])();
        this->y = extract< float >(position[1])();
    } catch (...) {
        PyErr_Print();
    }
    resize_event (_x,_y,_w,_h);
}

//_________________________________________________________________get_position
object
Viewport::get_position (void)
{
    return make_tuple (x,y);
}

//_____________________________________________________________________set_size
void
Viewport::set_size (object size)
{
    try {
        this->w = extract< float >(size[0])();
        this->h = extract< float >(size[1])();
    } catch (...) {
        PyErr_Print();
    }
    resize_event (_x,_y,_w,_h);
}

//_____________________________________________________________________get_size
object
Viewport::get_size (void)
{
    return make_tuple (w,h);
}

//_________________________________________________________________get_geometry
object
Viewport::get_geometry (void)
{
    return make_tuple (geometry[0], geometry[1], geometry[2], geometry[3]);
}

//_________________________________________________________________select_event
void
Viewport::select_event (int x, int y)
{
    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);

    observer->select_event(x,y);

    if (mode == GL_SELECT) {
        for (unsigned int i=0; i<viewports.size(); i++)
            viewports[i]->select_event (x,y);
        return;
    }  
    
    GLuint buffer[512];
    GLint hits;

    glSelectBuffer (512, buffer);
    glRenderMode (GL_SELECT);
    glInitNames ();
    glPushName (0);

    glPushMatrix();
    glLoadIdentity();
    render();
    hits = glRenderMode (GL_RENDER);

    if (hits > 0) {
        int choose = buffer[3];
        int depth = buffer[1];	
        for (int loop=1; loop<hits; loop++) {
            if (buffer[loop*4+1] < GLuint(depth)) {
                choose = buffer[loop*4+3];
                depth = buffer[loop*4+1];
            }
        }
        select (choose);
    } else {
        select (0);
    }

    glPopMatrix();
    glPopName();
}

//__________________________________________________________________focus_event
void
Viewport::focus_event (int x, int y)
{
    if ((has_focus) && (observer->button))
        return;

    if ((x > geometry[0]) && (x < (geometry[0]+geometry[2])) &&
        (y > geometry[1]) && (y < (geometry[1]+geometry[3]))) {
        has_focus = true;
        child_has_focus = false;        

        if (observer->button)
            return ;

        for (unsigned int i=0; i<viewports.size(); i++) {
            viewports[i]->focus_event (x, y);
            if (viewports[i]->observer->button) {
                child_has_focus = true;
                return;
            }
            // We tolerate only one child focused viewport
            if (viewports[i]->has_focus) {
                child_has_focus = true;
//                if (i > 0)            
//                   viewports[i]->has_focus = false;
            }
        }
    } else {
        has_focus = false;
        child_has_focus = false;
        for (unsigned int i=0; i<viewports.size(); i++)
            viewports[i]->has_focus = false;
        return;    
    }
}

//_________________________________________________________________resize_event
void
Viewport::resize_event (int x, int y, int w, int h)
{
    _x = x;
    _y = y;
    _h = h;
    _w = w;

    // Compute size
    int X, Y, W, H;

    if (this->w > 1.0f)
        W = int(this->w);
    else
        W = int(this->w * w);

    if (this->h > 1.0f)
        H = int(this->h);
    else
        H = int(this->h * h);

    if (W < 5)  W = 5;
    if (H < 5)  H = 5;
        
    // Compute position
    if (this->x < 0)
        if (this->x <= -1)
            X = x + int (w + this->x + 1 - W);
        else
            X = x + int (w + this->x*w + 1 - W);
    else
        if (this->x >= 1)
            X = x + int (this->x);
        else
            X = x + int (this->x*w);

    if (this->y < 0)
        if (this->y <= -1)
            Y = y + int (h + this->y + 1 - H);
        else
            Y = y + int (h + this->y*h + 1 - H);
    else
        if (this->y >= 1)
            Y = y + int (this->y);
        else
            Y = y + int (this->y*h);

    if (X < x)        X = x;
    if (X >= (x+w-1)) X = x+w-1;
    if (Y < y)        Y = y;
    if (Y >= (y+h-1)) Y = y+h-1;

    geometry[0] = X;
    geometry[1] = Y;
    geometry[2] = W;
    geometry[3] = H;

    observer->resize_event (X,Y,W,H);
    for (unsigned int i = 0; i<viewports.size(); i++)
        viewports[i]->resize_event (X,Y,W,H);

    for (unsigned int i = 0; i<objects.size(); i++)
        objects[i]->dirty = true;
}

//______________________________________________________________key_press_event
void
Viewport::key_press_event (std::string key)
{
    if ((!visible) || (!has_focus))
        return;
}

//____________________________________________________________key_release_event
void
Viewport::key_release_event (void)
{
    if ((!visible) || (!has_focus))
        return;
}

//___________________________________________________________button_press_event
void
Viewport::button_press_event (int button, int x, int y)
{
    if ((!visible) || (!has_focus))
        return;
        
    if (!child_has_focus)
        observer->button_press_event (button,x,y);
    else {
        observer->button = 0;
        for (unsigned int i=0; i<viewports.size(); i++)
            if (viewports[i]->has_focus)
                viewports[i]->button_press_event (button, x, y);
    }
}

//_________________________________________________________button_release_event
void
Viewport::button_release_event (int button, int x, int y)
{
    if ((!visible) || (!has_focus))
        return;

    focus_event(x,y);
    if (!child_has_focus) {
        observer->button_release_event (button,x,y);
        observer->button = 0;
    } else {
        observer->button = 0;
        for (unsigned int i=0; i<viewports.size(); i++) {
            if (viewports[i]->has_focus)
                viewports[i]->button_release_event (button, x, y);
            viewports[i]->observer->button = 0;
        }
    }
}

//_________________________________________________________pointer_motion_event
void
Viewport::pointer_motion_event (int x, int y)
{
    if ((!visible) || (!has_focus))
        return;
    if (!child_has_focus) {
        observer->pointer_motion_event (x, y);
    } else {
        for (unsigned int i=0; i<viewports.size(); i++) {
            if (viewports[i]->has_focus)
                viewports[i]->pointer_motion_event (x, y);
        }
    }
}

//________________________________________________________________python_export
void
Viewport::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Viewport> >();

    // member function pointers for overloading
    void (Viewport::*append_object)(ObjectPtr) = &Viewport::append;
    void (Viewport::*append_viewport)(ViewportPtr) = &Viewport::append;


    class_<Viewport, bases<Object> > ("Viewport",
    "======================================================================\n"
    "                                                                      \n"
    "A viewport represents a rectangular sub-area of the display and is    \n"
    "meant to be independent of other viewports. It possesses its own obs- \n"
    "-erver, list of  objects and if it has focus, it receives and process \n"
    "events. Size and position of the viewport can be in absolute (> 1) or \n"
    "relative coordinates (< 1).                                           \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "   observer   - observer looking at the viewport                      \n"
    "   has_focus  - whether viewport has focus                            \n"
    "   has_border - whether viewport has surrounding border               \n"
    "   is_ortho   - whether viewport use ortho projection mode            \n"
    "   size       - size request                                          \n"
    "   position   - position request                                      \n"
    "   geometry   - actual window relative geometry as a (x,y,w,h) tuple  \n"
    "                                                                      \n"
    "======================================================================\n",
    init< optional <tuple, tuple, bool, bool, std::string> > (
        (arg("size") = make_tuple (1.0f,1.0f),
         arg("position") = make_tuple (0.0f,0.0f),
         arg("has_border") = false,
         arg("is_ortho") = false,
         arg("name") = "Viewport"),
        "__init__ (size, position, has_border, name )\n"))

    .def_readonly ("observer", &Viewport::observer)
    .def_readonly ("has_focus", &Viewport::has_focus)
    .def_readwrite("has_border", &Viewport::has_border)
    .add_property ("is_ortho", &Viewport::get_is_ortho,&Viewport::set_is_ortho)
    .add_property ("size", &Viewport::get_size,&Viewport::set_size)
    .add_property ("position", &Viewport::get_position,&Viewport::set_position)
    .add_property ("geometry", &Viewport::get_geometry)

    .def ("append", append_object)
    .def ("append", append_viewport,
          "append (o)\n\n"
          "Append object to viewport.")

    .def ("clear", &Viewport::clear,
            "clear()\n"
            "Remove all objects from the viewport")

    .def ("__getitem__", &Viewport::getitem,
            "x.__getitem__ (y)  <==> x[y]\n")

    .def ("__delitem__", &Viewport::delitem,
            "x.__delitem__ (y)  <==> x[y]\n")

    .def ("__len__", &Viewport::len,
            "__len__() -> integer\n")

    .def ("select_event", &Viewport::select_event,
          "select_event (x,y)")

    .def ("focus_event", &Viewport::focus_event,
          "focus_event (x,y)")

    .def ("resize_event", &Viewport::resize_event,
          "resize_event (x,y,w,h)")

    .def ("key_press_event", &Viewport::key_press_event,
          "key_press_event (key)")

    .def ("key_release_event", &Viewport::key_release_event,
          "key_release_event ()")

    .def ("button_press_event", &Viewport::button_press_event,
          "button_press_event (button, x, y)")
 
    .def ("button_release_event", &Viewport::button_release_event,
          "button_release_event (button,x,y)")
 
    .def ("pointer_motion_event", &Viewport::pointer_motion_event,
          "pointer_motion_event (x,y)")
 
    ;       
}
