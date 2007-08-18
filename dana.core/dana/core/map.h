//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: map.h 249 2007-07-19 15:30:16Z rougier $

#ifndef __DANA_CORE_MAP_H__
#define __DANA_CORE_MAP_H__

#include <boost/thread/barrier.hpp>
#include <vector>
#include "object.h"
#include "layer.h"
#include "unit.h"


namespace dana { namespace core {
    
    typedef boost::shared_ptr<class Map> MapPtr;

    class Map : public Object
    {
        public:
            //  attributes
            // ================================================================
            class Network *       network; // network owning this map
            std::vector<LayerPtr> layers;  // layers composing the map
            std::vector<std::vector<int> > shuffles;
            int                            shuffle_index;
            int                   width,height; // shape
            int                   x,y;     // position
            int                   dx, dy;  // offset
            int                   zoom;    // zoom
            object                frame;   // normalized position & shape
            static unsigned long  epochs; // proxy epochs for thread evaluation
            static Map *          map;     // proxy map for thread evaluation
            SpecPtr               spec;    // specification for this map
            boost::barrier *      barrier; // thread synchronization barrier
            
        public:
            // life management 
            // ================================================================
            Map (object shape    = make_tuple(0,0),
                 object position = make_tuple(0,0,0,0,1));
            virtual ~Map (void);

            // content management
            // ================================================================
            virtual void       append (LayerPtr layer);
            virtual LayerPtr   get (int index);
            virtual int        size (void);
            
            // proxied management (default to layer 0)
            // ================================================================
            virtual UnitPtr    unit (int index);
            virtual UnitPtr    unit (int x, int y);
            virtual int        fill (object type);
            virtual object     get_potentials (void);


            // activity management
            // ================================================================
            static void        evaluate  (void);
            virtual void       clear (void);
            virtual void       compute_dp  (void);
            virtual void       compute_dw  (void);
            

            //  attribute manipulation
            // ===========r====================================================
            virtual SpecPtr    get_spec (void);
            virtual void       set_spec (SpecPtr s);  
            virtual object     get_shape (void);
            virtual void       set_shape (object shape);
            virtual void       set_shape (int w, int h);
            virtual object     get_position (void);
            virtual void       set_position (object position);
            virtual void       set_position (int x, int y);
            virtual object     get_frame (void);
            virtual void       set_frame (object frame);

        public:
            // python export
            // ================================================================
            static void         boost (void);
    };

}} // namespace dana::core
#endif
