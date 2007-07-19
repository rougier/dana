//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __DANA_CORE_UNIT_H__
#define __DANA_CORE_UNIT_H__

#include <boost/python.hpp>
#include <vector>
#include "object.h"
#include "layer.h"
#include "link.h"
#include "spec.h"

using namespace boost::python;


namespace dana { namespace core {

    typedef boost::shared_ptr<class Layer> LayerPtr;
    typedef boost::shared_ptr<class Link> LinkPtr;    
    typedef boost::shared_ptr<class Unit> UnitPtr;

    class Unit : public Object {
        public:
            //  attributes
            // ================================================================
            class Layer *        layer;     // layer owning this unit
            float                potential; // potential
            std::vector<LinkPtr> laterals;  // lateral links
            std::vector<LinkPtr> afferents; // afferent links
            SpecPtr              spec;      // specification of the unit
            int                  x, y;      // position within layer

        public:
            //  life management
            // ================================================================
            Unit(void);
            virtual ~Unit(void);
            
            //  content management
            // ================================================================
            virtual void connect (UnitPtr source, float weight, object data);            
            virtual void connect (UnitPtr source, float weight);
            virtual void connect (LinkPtr link);
            virtual void clear (void);

            //  object management
            // ================================================================
            virtual float       compute_dp (void);
            virtual float       compute_dw (void);

            //  attribute manipulation
            // ================================================================
            virtual LayerPtr    get_layer (void) const;
            virtual void        set_layer (class Layer *l);
            virtual int         get_x (void) const;
            virtual void        set_x (const int value);
            virtual int         get_y (void) const;
            virtual void        set_y (const int value);
            virtual SpecPtr     get_spec (void) const;
            virtual void        set_spec (const SpecPtr s);            
            virtual object      get_position (void) const;
            virtual void        set_position (const object p);
            virtual void        set_position (const int x, const int y);
            
            // convenient methods
            // ================================================================
            virtual object      get_weights  (const LayerPtr layer);

            // python export
            // ================================================================
            static void boost (void);
    };

}} // namespace dana::core

#endif
