//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
// 02111-1307, USA.
// $Id$

#ifndef __DANA_SIGMAPI_LINK_H__
#define __DANA_SIGMAPI_LINK_H__

#include <boost/python.hpp>
#include "core/link.h"
#include "core/object.h"
#include "unit.h"
#include <vector>

inline float MIN(float x,float y) { if(x < y) return x; return y;}
inline float MAX(float x,float y) { if(x < y) return y; return x;}

using namespace boost::python;

namespace dana { namespace sigmapi {

    typedef enum 
        {
            SIGMAPI_MAX,
            SIGMAPI_PROD
        }
    LinkType;


    class Link : public core::Link {
    public:
        std::vector<core::UnitPtr> source;
        float	weight;
        LinkType type;

    public:
        Link (LinkType t);
        virtual ~Link (void);

        core::UnitPtr get_source (const int i) const;
        void	add_source (const core::UnitPtr src);
        float	get_weight (void) const;
        void	set_weight (const float w);

        float compute(void);
    public:
        static void	boost (void);
    };

}} // namespace dana::core

#endif

