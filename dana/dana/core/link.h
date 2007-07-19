//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_LINK_H__
#define __DANA_CORE_LINK_H__

#include "object.h"
#include "unit.h"

using namespace boost::python;

namespace dana { namespace core {

    typedef boost::shared_ptr<class Unit>  UnitPtr;
    typedef boost::shared_ptr<class Link>  LinkPtr;
    
    class Link {
    public:
        UnitPtr	source;
        float	weight;

    public:
        Link (void);
        Link (UnitPtr const src, float const w=0.0f);
        virtual ~Link (void);
    
        UnitPtr         get_source (void) const;
        void	        set_source (const UnitPtr src);
        float	        get_weight (void) const;
        void	        set_weight (const float w);
        virtual float   compute ();
        static void	    python_export (void);
    };

}}

#endif

