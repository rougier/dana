//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$

#ifndef __DANA_SIGMAPI_UNIT_H__
#define __DANA_SIGMAPI_UNIT_H__

#include <boost/python.hpp>
#include "core/unit.h"
#include "link.h"

using namespace boost::python;

namespace dana { namespace sigmapi {
    //typedef boost::shared_ptr<class Link> LinkPtr;
   //typedef boost::shared_ptr<class Unit> UnitPtr;
    // Unit class
    class Unit : public core::Unit {

    public:
        std::vector<core::LinkPtr> afferents;
        std::vector<core::LinkPtr> laterals;

        public:
            // Constructor
            Unit(void);

            // Desctructor
            virtual ~Unit(void);

	    //virtual void connect (core::UnitPtr source, float weight);
            void connect (core::LinkPtr link);
            // Evaluate new potential and return difference
            virtual float compute_dp (void);

        public:
            // Boost python extension
            static void boost (void);
    };

}} // namespace dana::cnft

#endif
