//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_LAYER_H__
#define __DANA_CORE_LAYER_H__

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <vector>
#include "object.h"

using namespace boost::python;


namespace dana { namespace core {

    class Layer : public Object {
        public:
            //  attributes
            // ================================================================
            class Map *          map;       // map owning this layer
            std::vector<UnitPtr> units;     // units composing the layer
            std::vector<UnitPtr> permuted;  // permuted units
            object               spec;      // specification of the layer

        public:
            // life management
            // =================================================================
            Layer (void);
            virtual ~Layer (void);

            // content management
            // =================================================================
            virtual void         append (UnitPtr unit);
            virtual UnitPtr      get (const int index) const;
            virtual UnitPtr      get (const int x, const int y) const;
            virtual int          size (void) const;
            virtual int          fill (object type);

            // activity management
            // =================================================================
            virtual void         clear (void);
            virtual float        compute_dp (void);
            virtual float        compute_dw (void);

            //  attribute manipulation
            // =================================================================
            virtual Map *        get_map (void);
            virtual void         set_map (class Map *m);
            virtual object       get_spec (void) const;
            virtual void         set_spec (const object s);

            // convenient methods
            // =================================================================
            virtual object       get_potentials (void) const;

            // python export
            // =================================================================
            static void          boost (void);
    };

}} // namespace dana::core

#endif
