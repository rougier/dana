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

#ifndef __GLPYTHON_CORE_OBJECT_H__
#define __GLPYTHON_CORE_OBJECT_H__

#include <string>
#include <boost/python.hpp>

namespace py = boost::python;

namespace dana {
    namespace figure {
        // Structure to hold 4 floats indifferently as:
        //   - an array of 4 floats
        //   - x,y,z,w tuple
        //   - dx, dy, dz, dw tuple
        //   - r,g,b,a tuple
        //   - red, green, blue, alpha tuple
        //   - up, down, left, right
        struct Vec4f {
            union {
                float data[4];
                struct {
                    union {float x; float dx; float r; float red;   float up;};
                    union {float y; float dy; float g; float green; float down;};
                    union {float z; float dz; float b; float blue;  float left;};
                    union {float w; float dw; float a; float alpha; float right;};
                };
            };
            Vec4f (float a=0, float b=0, float c=0, float d=1) : x(a), y(b), z(c), w(d)
            {};
            Vec4f (float d[4])
            {
                data[0] = d[0]; data[1] = d[1]; data[2] = d[2]; data[3] = d[3];
            };
            Vec4f (py::tuple &t)
            {
                int size = py::extract< int > (t.attr("__len__")());
                try {
                    x = py::extract< float >(t[0])();
                    y = py::extract< float >(t[1])();
                    z = 0;
                    w = 1;
                    if (size > 2)
                        z = py::extract< float >(t[2])();
                    if (size > 3)
                        w = py::extract< float >(t[3])();
                } catch (...) {
                    PyErr_SetString(PyExc_AssertionError, "tuple of 2, 3 or 4 floats expected");
                    py::throw_error_already_set();
                }
            }
        };
        typedef Vec4f Position;
        typedef Vec4f Margin;
        typedef Vec4f Shape;
        typedef Vec4f Size;

        // Object class
        typedef boost::shared_ptr<class Object> ObjectPtr;
        class Object {
            
        public:
            std::string name_;        // name
            bool visible_;            // whether to render object
            Position position_;       // position
            Size size_;               // size
            int id_;                  // id for selection
            static int id_counter_;
            
        public:
            Object (void);
            virtual ~Object (void);
            
            virtual void build  (void);
            virtual void render (void);
            virtual void update (void);
            
            virtual void        set_name (std::string name);
            virtual std::string get_name (void);
            virtual std::string     name (void);

            virtual void set_visible (bool visible);
            virtual bool get_visible (void);
            virtual bool     visible (void);

            virtual void      set_position (Position position);
            virtual void      set_position_tuple (py::tuple position);
            virtual Position  get_position (void);
            virtual py::tuple get_position_tuple (void);
            virtual Position      position (void);

            virtual void      set_size (Size size);
            virtual void      set_size_tuple (py::tuple size);
            virtual Size      get_size (void);
            virtual py::tuple get_size_tuple (void);
            virtual Size          size (void);

            virtual void set_id (int id);
            virtual int  get_id (void);
            virtual int      id (void);
            
            static void python_export (void);
        };

    } // namespace core

} // namespace glpython

#endif
