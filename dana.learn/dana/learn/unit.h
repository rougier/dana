//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_LEARN_UNIT_H__
#define __DANA_LEARN_UNIT_H__

#include <boost/python.hpp>
#include "cnft/unit.h"
#include "core/layer.h"
#include "core/object.h"

using namespace boost::python;

namespace dana { namespace learn {

    class Unit : public cnft::Unit {

        public:
            //  attributes
            // =================================================================
	    std::vector<std::vector<float> > * learnFunc;

        public:
            //  life management
            // =================================================================
        Unit(void);
        virtual ~Unit(void);

            //  object management
            // =================================================================
	    void set_learning_rule(std::vector< std::vector<float> > * learnFunc);
	    void learn(core::LayerPtr src,float scale);

        public:
            // python export
            // =================================================================
            static void boost (void);
    };
}} // namespace dana::learn

#endif
