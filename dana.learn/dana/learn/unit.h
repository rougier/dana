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
#include "dana/cnft/unit.h"
#include "dana/core/layer.h"
#include "dana/core/object.h"

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
	    virtual void set_learning_rule(std::vector< std::vector<float> > * learnFunc);
	    virtual void learn(core::LayerPtr src,float scale);
        virtual float find_weight_with(int mpos_x,int mpos_y,int x,int y);
        
        public:
            // python export
            // =================================================================
            static void boost (void);
    };
}} // namespace dana::learn

#endif
