//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_LEARN_SUNIT_H__
#define __DANA_LEARN_SUNIT_H__

#include <boost/python.hpp>
#include "dana/cnft/unit.h"
#include "dana/core/layer.h"
#include "dana/core/object.h"
#include "unit.h"

using namespace boost::python;

namespace dana { namespace learn {

    class SUnit : public learn::Unit {

        public:
            //  life management
            // =================================================================
        SUnit(void);
        virtual ~SUnit(void);
        virtual float compute_dp (void);
        public:
        // python export
        // =================================================================
        static void boost (void);
    };
}} // namespace dana::learn

#endif
