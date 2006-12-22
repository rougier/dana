//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_CORE_MAP_H__
#define __DANA_CORE_MAP_H__

#include <boost/python.hpp>
#include <vector>
#include "object.h"

using namespace boost::python;


namespace dana { namespace core {
    
    // Map class
    class Map : public Object
    {
        public:
            //  attributes
            // =================================================================
            class Network *       network; // network owning this map
            std::vector<LayerPtr> layers;  // layers composing the map
            std::vector<std::vector<int> > shuffles;
            int                     shuffle_index;
            int          x,y,width,height; // position & shape
            object                frame;   // normalized position & shape
            static unsigned long  epochs;  // proxy epochs for thread evaluation
            static Map *          map;     // proxy map for thread evaluation
            object                spec;    // specification for this map
            
        public:
            // life management 
            // =================================================================
            Map (object shape=make_tuple(0,0), object position=make_tuple(0,0));
            virtual ~Map (void);

            // content management
            // =================================================================
            virtual void         append (LayerPtr layer);
            virtual LayerPtr     get (const int index) const;
            virtual int          size (void) const;
            virtual void         clear (void);
            
            // proxied management (default to layer 0)
            // =================================================================
            virtual UnitPtr      unit (const int index) const;                     
            virtual UnitPtr      unit (const int x, const int y) const;
            virtual int          fill (object type);
            virtual object       get_potentials (void) const;


            // activity management
            // =================================================================
            virtual void        evaluate  (void);
            static void         static_evaluate (void);

            //  attribute manipulation
            // =================================================================
            virtual object      get_spec (void) const;
            virtual void        set_spec (const object s);  
            virtual object      get_shape (void) const;
            virtual void        set_shape (const object shape);
            virtual void        set_shape (const int w, const int h);
            virtual object      get_position (void) const;
            virtual void        set_position (const object position);
            virtual void        set_position (const int x, const int y);
            virtual object      get_frame (void) const;
            virtual void        set_frame (const object frame);

        public:
            // python export
            // =================================================================
            static void         boost (void);
    };

}} // namespace dana::core
#endif
