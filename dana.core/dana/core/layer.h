//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: layer.h 257 2007-07-29 11:38:44Z rougier $

#ifndef __DANA_CORE_LAYER_H__
#define __DANA_CORE_LAYER_H__

#include <vector>
#include "object.h"
#include "unit.h"
#include "link.h"
#include "spec.h"


namespace dana { namespace core {

    typedef boost::shared_ptr<class Unit> UnitPtr;
    typedef boost::shared_ptr<class Layer> LayerPtr;

    // ______________________________________________________________class Layer
    class Layer : public Object {
    public:
        // ___________________________________________________________attributes
        class Map *          map; 
        std::vector<UnitPtr> units;
        std::vector<UnitPtr> permuted;
        SpecPtr              spec;
        py::object           potentials;

    public:
        // _________________________________________________________________life
        Layer (void);
        virtual ~Layer (void);

        // _________________________________________________________________main
        virtual void         append (UnitPtr unit);
        virtual UnitPtr      get (const int index) const;
        virtual UnitPtr      get (const int x, const int y) const;
        virtual int          size (void) const;
        virtual int          fill (py::object type);
        virtual void         clear (void);
        virtual float        compute_dp (void);
        virtual float        compute_dw (void);

        // __________________________________________________________________I/O
        virtual int write (xmlTextWriterPtr writer);
        virtual int read  (xmlTextReaderPtr reader);

        // ______________________________________________________________get/set
        virtual Map *        get_map (void);
        virtual void         set_map (class Map *map);
        virtual SpecPtr      get_spec (void) const;
        virtual void         set_spec (const SpecPtr spec);
        virtual py::object   get_potentials (void);
        virtual void         set_potentials (numeric::array a);

        // _______________________________________________________________export
        static void          python_export (void);
    };

}} // namespace dana::core

#endif
