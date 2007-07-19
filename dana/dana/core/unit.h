//
// Copyright (C) 2006,2007 Nicolas Rougier
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

    class Unit : public Object
    {
        public:
            class Layer *        layer;
            float                potential;
            std::vector<LinkPtr> laterals;
            std::vector<LinkPtr> afferents;
            SpecPtr              spec;
            int                  x, y;

        public:
            Unit (float potential = 0.0f);
            virtual ~Unit(void);

            virtual void        connect (UnitPtr source, float weight, object data);
            virtual void        connect (UnitPtr source, float weight);
            virtual void        connect (LinkPtr link);
            virtual void        clear (void);
            virtual float       compute_dp (void);
            virtual float       compute_dw (void);

            virtual LayerPtr    get_layer (void);
            virtual void        set_layer (class Layer *layer);
            virtual int         get_x (void);
            virtual void        set_x (int x);
            virtual int         get_y (void);
            virtual void        set_y (int y);
            virtual SpecPtr     get_spec (void);
            virtual void        set_spec (SpecPtr s);
            virtual tuple       get_position (void);
            virtual void        set_position (tuple position);
            virtual void        set_position (int x, const int y);
            virtual object      get_weights  (LayerPtr layer);

            static void         python_export (void);
    };
}}

#endif
