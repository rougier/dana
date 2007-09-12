//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix 
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_SVD_UNIT_H__
#define __DANA_SVD_UNIT_H__

#include <boost/python.hpp>
#include "dana/cnft/unit.h"

using namespace boost::python;

namespace dana { namespace svd {

    class Unit : public cnft::Unit {
        public:
            //  life management
            // =================================================================
            Unit(void);
            virtual ~Unit(void);

            // convenient methods
            // ================================================================
            virtual object      get_weights  (const core::LayerPtr layer);
            
        public:
            // python export
            // =================================================================
            static void boost (void);
    };
}} // namespace dana::svd

#endif
