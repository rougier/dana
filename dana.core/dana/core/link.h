//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: link.h 244 2007-07-19 10:09:40Z rougier $
// _____________________________________________________________________________

#ifndef __DANA_CORE_LINK_H__
#define __DANA_CORE_LINK_H__

#include "object.h"

namespace py = boost::python;

namespace dana { namespace core {

    typedef boost::shared_ptr<class Unit>  UnitPtr;
    typedef boost::shared_ptr<class Link>  LinkPtr;

    // _______________________________________________________________class Link
    class Link {
        // ___________________________________________________________attributes
    public:
        UnitPtr	          source;
        float	          weight;

    public:
        // _________________________________________________________________life
        Link (void);
        Link (UnitPtr source, float weight=0.0f);
        virtual ~Link (void);

        // _________________________________________________________________main
        virtual float              compute (void);

        // __________________________________________________________________I/O
        virtual int write (xmlTextWriterPtr writer);
        virtual int read  (xmlTextReaderPtr reader);

        // ______________________________________________________________get/set
        virtual py::tuple    const get (void);
        virtual UnitPtr      const get_source (void);
        virtual void	           set_source (UnitPtr src);
        virtual float        const get_weight (void);
        virtual void	           set_weight (float w);

        // _______________________________________________________________export
        static void                python_export (void);
    };

}}

#endif

