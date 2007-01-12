//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_LIFE_UNIT_H__
#define __DANA_LIFE_UNIT_H__

#include <boost/python.hpp>
#include "core/unit.h"

using namespace boost::python;

namespace dana { namespace life {
    // Unit class
    class Unit : public core::Unit {
        public:
            //  attributes
            // =================================================================
            float _potential;

        public:
            //  life management
            // =================================================================
            Unit(void);
            virtual ~Unit(void);
            
            //  object management
            // =================================================================
            virtual float compute_dp (void);
            virtual float compute_dw (void);
            
        public:
            // python export
            // =================================================================        
            static void boost (void);
    };

}} // namespace dana::life

#endif
