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

#ifndef __DANA_FIGURE_COLORMAP_H__
#define __DANA_FIGURE_COLORMAP_H__

#include <vector>
#include <boost/python.hpp>
#include "color.h"

namespace py = boost::python;


namespace dana {
    namespace figure {

        typedef boost::shared_ptr<class Colormap> ColormapPtr;

        class Colormap {
        public:
            std::vector<Color>  colors;
            std::vector<Color>  samples;
            float               min, max;
            int                 resolution;
            
        public:
            Colormap                 (void);
            Colormap                 (const Colormap &other);            
            ~Colormap                (void);
            
            std::string repr         (void);
            
            unsigned int len         (void);
            void         clear       (void);
            void         append      (float value, py::object color);
            Color        get         (int index);
            Color        color       (float value);
            Color        exact_color (float value);            
            void         scale       (float inf, float sup);
            void         sample      (void);

            static void  python_export (void);
        };
    } // namespace dana
} // namespace figure

#endif
