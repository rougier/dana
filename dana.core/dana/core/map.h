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

    // ________________________________________________________________class Map
    class Map : public Object {

    public:
        // ___________________________________________________________attributes
        class Network *       network; // network owning this map
        std::vector<LayerPtr> layers;  // layers composing the map
        std::vector<std::vector<int> > shuffles;
        int                            shuffle_index;
        int                   width,height; // shape
        int                   x,y;     // position
        int                   dx, dy;  // offset
        int                   zoom;    // zoom
        py::object                frame;   // normalized position & shape
        static unsigned long  epochs; // proxy epochs for thread evaluation
        static Map *          map;     // proxy map for thread evaluation
        SpecPtr               spec;    // specification for this map
        boost::barrier *      barrier; // thread synchronization barrier
            
        public:
        // _________________________________________________________________life
        Map (py::object shape    = py::make_tuple(0,0),
             py::object position = py::make_tuple(0,0,0,0,1));
        virtual ~Map (void);

        // _________________________________________________________________main
        virtual void       append (LayerPtr layer);
        virtual LayerPtr   get (int index);
        virtual int        size (void);        
        virtual UnitPtr    unit (int index);
        virtual UnitPtr    unit (int x, int y);
        virtual int        fill (py::object type);
        static void        evaluate  (void);
        virtual void       clear (void);
        virtual void       compute_dp  (void);
        virtual void       compute_dw  (void);
            
        // __________________________________________________________________I/O
        virtual int write (xmlTextWriterPtr writer);
        virtual int read  (xmlTextReaderPtr reader);

        // ______________________________________________________________get/set
        virtual SpecPtr    get_spec (void);
        virtual void       set_spec (SpecPtr s);  
        virtual py::object get_shape (void);
        virtual void       set_shape (py::object shape);
        virtual void       set_shape (int w, int h);
        virtual py::object get_position (void);
        virtual void       set_position (py::object position);
        virtual void       set_position (int x, int y);
        virtual py::object get_frame (void);
        virtual void       set_frame (py::object frame);
        virtual py::object get_potentials (void);

        // _______________________________________________________________export
        static void        python_export (void);
    };

}} // namespace dana::core
#endif
