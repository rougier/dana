// -----------------------------------------------------------------------------
// DANA 
// Copyright (C) 2006-2007  Nicolas P. Rougier
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

#ifndef __DANA_FIGURE_ARRAY_H__
#define __DANA_FIGURE_ARRAY_H__

#include <GL/gl.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "../../object.h"
#include "../../colormap.h"
#include "../../colormaps.h"

namespace numpy = boost::python::numeric;
namespace py = boost::python;


namespace dana {
    namespace figure {

    typedef boost::shared_ptr<class Array> ArrayPtr;

        class Array : public Object {
        public:
            PyArrayObject *     array_;
            float *             data_;
            Shape               shape_;
            ColormapPtr         cmap_;
            GLuint              tex_id_;

        public:
            Array (numpy::array X, ColormapPtr cmap, py::tuple position, py::tuple size);
            virtual ~Array(void);
            virtual void render  (void);
            virtual void update  (void);
            virtual void         set_data (numpy::array X);
            virtual numpy::array get_data (void);

            static void python_export (void);
        };
    } // namespace dana
} // namespace figure

#endif
