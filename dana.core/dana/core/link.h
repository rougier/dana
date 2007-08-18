//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: link.h 244 2007-07-19 10:09:40Z rougier $

#ifndef __DANA_CORE_LINK_H__
#define __DANA_CORE_LINK_H__

#include "object.h"

namespace dana { namespace core {

    typedef boost::shared_ptr<class Unit>  UnitPtr;
    typedef boost::shared_ptr<class Link>  LinkPtr;
    
    class Link {
    public:
        UnitPtr	source;
        float	weight;

    public:
        Link (void);
        Link (UnitPtr source, float weight=0.0f);
        virtual ~Link (void);
    
        UnitPtr         get_source (void);
        void	        set_source (UnitPtr src);
        float	        get_weight (void);
        void	        set_weight (float w);
        virtual float   compute (void);
        static void	    python_export (void);
    };

}}

#endif

