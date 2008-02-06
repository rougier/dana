// -----------------------------------------------------------------------------
// glpython - an OpenGL terminal
// Copyright (C) 2007  Nicolas P. Rougier
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

#include "object.h"
#include <GL/gl.h>

using namespace dana::figure;


int Object::id_counter_ = 1;

Object::Object (void)
{
    set_visible (true);
    set_name ("Object");
    set_position (Position (0,0,0));
    set_size (Size (1,1,1));
    set_id (id_counter_++);
}

Object::~Object (void)
{}

void
Object::build (void)
{}

void
Object::render (void)
{}

void
Object::update (void)
{}

void
Object::set_name (std::string name)
{
    name_ = name;
}

std::string Object::get_name (void)
{
    return name_;
}

std::string
Object::name (void)
{
    return get_name();
}

void
Object::set_position (Position position)
{
    position_ = Position (position);
}
void
Object::set_position_tuple (py::tuple position)
{
    position_ = Position (position);
}
Position
Object::get_position (void)
{
    return position_;
}
py::tuple
Object::get_position_tuple (void)
{
    return py::make_tuple (position_.x, position_.y, position_.z);
}
Position
Object::position (void)
{
    return get_position();
}

void
Object::set_size (Size size)
{
    size_ = Size (size);
}
void
Object::set_size_tuple (py::tuple size)
{
    size_ = Size (size);
}
Size
Object::get_size (void)
{
    return size_;
}
py::tuple
Object::get_size_tuple (void)
{
    return py::make_tuple (size_.x, size_.y, size_.z);
}
Size
Object::size (void)
{
    return get_size();
}

void
Object::set_visible (bool visible)
{
    visible_ = visible;
}

bool
Object::get_visible (void)
{
    return visible_;
}

bool
Object::visible (void)
{
    return get_visible();
}

void Object::set_id (int id)
{
    id_ = id;
}

int
Object::get_id (void)
{
    return id_;
}

int
Object::id (void)
{
    return get_id();
}


void
Object::python_export (void) {

    using namespace boost::python;
    py::docstring_options doc_options;
    doc_options.disable_signatures();

    register_ptr_to_python< boost::shared_ptr<Object> >();
    class_<Object>
        ("Object",
         " Object                                                                \n"
         "                                                                       \n"
         " ______________________________________________________________________\n"
         "                                                                       \n"
         " Object is the base class for any renderable object.                   \n"
         "                                                                       \n"
         " Object attributes:                                                    \n"
         "                                                                       \n"
         "    name -- name of the object                                         \n"
         "    visible -- visibility status                                       \n"
         "    size -- size                                                       \n"
         "    position -- position                                               \n"
         "    id -- identification (for selection operations)                    \n"
         "                                                                       \n"
         " ______________________________________________________________________\n",

         init < > ("Create a new renderable object"))

        .add_property ("visible",
                       &Object::get_visible, &Object::set_visible, 
                       "visibility status")
        .add_property ("name",
                       &Object::get_name, &Object::set_name, 
                       "name of the object")
        .add_property ("id",
                       &Object::get_id,
                       "object unique identification")
        .add_property ("size",
                       &Object::get_size_tuple, &Object::set_size_tuple, 
                       "size of the object")
        .add_property ("position",
                       &Object::get_position_tuple, &Object::set_position_tuple, 
                       "position of the object")
        
        .def ("build", &Object::build,
              "build()\n"
              "build the object.")
        .def ("render", &Object::render,
              "render()\n"
              "render the object.")
        .def ("update", &Object::update,
              "update()\n"
              "update the object.")
    ;       
}
