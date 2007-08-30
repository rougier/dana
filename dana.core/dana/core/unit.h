//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: unit.h 246 2007-07-19 10:19:18Z rougier $
// ____________________________________________________________________________

#ifndef __DANA_CORE_UNIT_H__
#define __DANA_CORE_UNIT_H__

#include <boost/python.hpp>
#include <vector>
#include "object.h"
#include "layer.h"
#include "link.h"
#include "spec.h"

namespace dana { namespace core {

    typedef boost::shared_ptr<class Layer> LayerPtr;
    typedef boost::shared_ptr<class Link> LinkPtr;    
    typedef boost::shared_ptr<class Unit> UnitPtr;

    // _______________________________________________________________class Unit
    class Unit : public Object {

    public:
        // ___________________________________________________________attributes
        float                potential;
        SpecPtr              spec;
        std::vector<LinkPtr> laterals;
        std::vector<LinkPtr> afferents;
        class Layer *        layer;
        int                  x, y;

    public:
        // _________________________________________________________________life
        Unit (float potential = 0.0f);
        virtual ~Unit (void);

        // _________________________________________________________________main
        virtual float compute_dp (void);
        virtual float compute_dw (void);
        virtual void  connect    (UnitPtr source, float weight, py::object data);
        virtual void  connect    (UnitPtr source, float weight);
        virtual void  connect    (LinkPtr link);
        virtual void  clear      (void);

        // __________________________________________________________________I/O
        virtual int write (xmlTextWriterPtr writer);
        virtual int read  (xmlTextReaderPtr reader);
        
        // ______________________________________________________________get/set
        virtual py::object get_weights   (LayerPtr layer);
        virtual py::list   get_afferents (void);
        virtual py::list   get_laterals  (void);
        virtual float      get_potential (void);
        virtual void       set_potential (float potential);
        virtual SpecPtr    get_spec      (void);
        virtual void       set_spec      (SpecPtr spec);
        virtual LayerPtr   get_layer     (void);
        virtual void       set_layer     (class Layer *layer);
        virtual int        get_x         (void);
        virtual void       set_x         (int x);
        virtual int        get_y         (void);
        virtual void       set_y         (int y);
        virtual py::tuple  get_position  (void);
        virtual void       set_position  (py::tuple position);
        virtual void       set_position  (int x, int y);
        
        // ___________________________________________________________arithmetic
        virtual Unit &     operator= (const Unit &other);
        virtual Unit const operator+ (Unit const &other) const;
        virtual Unit const operator- (Unit const &other) const;
        virtual Unit const operator* (Unit const &other) const;
        virtual Unit const operator/ (Unit const &other) const;
        virtual Unit const operator+ (float value) const;
        virtual Unit const operator- (float value) const;
        virtual Unit const operator* (float value) const;
        virtual Unit const operator/ (float value) const;
        virtual Unit &     operator+= (Unit const &other);
        virtual Unit &     operator-= (Unit const &other);
        virtual Unit &     operator*= (Unit const &other);
        virtual Unit &     operator/= (Unit const &other);
        virtual Unit &     operator+= (float value);
        virtual Unit &     operator-= (float value);
        virtual Unit &     operator*= (float value);
        virtual Unit &     operator/= (float value);
        
        // _______________________________________________________________export
        static void  python_export (void);
    };
}}

#endif
