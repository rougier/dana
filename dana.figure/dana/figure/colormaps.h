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


#ifndef __DANA_FIGURE_COLORMAPS_H__
#define __DANA_FIGURE_COLORMAPS_H__

#include "colormap.h"

namespace dana {
    namespace figure {

        class Colormaps {
        public:
            static ColormapPtr Default;
            static ColormapPtr Ice;
            static ColormapPtr Fire;
            static ColormapPtr IceAndFire;
            static ColormapPtr Hot;
            static ColormapPtr Gray;
            static ColormapPtr Red;
            static ColormapPtr Green;
            static ColormapPtr Blue;

            static void make (void);
            static void python_export (void);
        };
    } // namespace dana
} // namespace figure

#endif
