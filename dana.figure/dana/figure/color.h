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

#ifndef __DANA_FIGURE_COLOR_H__
#define __DANA_FIGURE_COLOR_H__

#include <boost/python.hpp>

namespace py = boost::python;

namespace dana {
    namespace figure {

        const int RED   = 0;
        const int GREEN = 1;
        const int BLUE  = 2;
        const int ALPHA = 3;
        const int VALUE = 4;

        typedef boost::shared_ptr<class Color> ColorPtr;

        class Color {
        public:    
            float data[5];
            
        public:
            Color (float r=0, float g=0, float b=0, float a=1, float v=0);
            Color (py::tuple channels);
            Color (const Color &other);
            ~Color (void);

            void set        (int index, float r);
            void set_red    (float v);
            void set_green  (float v);
            void set_blue   (float v);
            void set_alpha  (float v);
            float get_red   (void);
            float get_green (void);
            float get_blue  (void);
            float get_alpha (void);
            float get_value (void);

            std::string repr (void);
            Color operator= (const Color &other);
            Color operator+ (const Color &other);
            Color operator* (const float scale);
            static bool cmp (Color c1, Color c2);

            static void python_export (void);
        };
    } // namespace dana
} // namespace figure

#endif
