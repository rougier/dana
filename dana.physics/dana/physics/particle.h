//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_PHYSICS_PARTICLE_H__
#define __DANA_PHYSICS_PARTICLE_H__

#include <boost/python.hpp>
#include "core/unit.h"

using namespace boost::python;

namespace dana { namespace physics {

    // Particle class
    class Particle : public core::Unit {

        public:
            //  life management
            // ================================================================        
            Particle (void);
            virtual ~Particle (void);
            
            //  object management
            // ================================================================
            virtual float compute_dp (void);
            
        public:
            // python export
            // ================================================================
            static void boost (void);
    };

}} // namespace dana::physics

#endif
