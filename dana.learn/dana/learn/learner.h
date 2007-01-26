//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_LEARN_LEARNER_H__
#define __DANA_LEARN_LEARNER_H__

#include <boost/python.hpp>
#include "unit.h"
#include "core/layer.h"
#include "core/object.h"
#include <numpy/arrayobject.h>

using namespace boost::python;

namespace dana { namespace learn {

	typedef struct {
		core::LayerPtr source;
		core::LayerPtr destination;
		std::vector<float> params;		
	} learnStr;
	
    class Learner {
	    private:
		    std::vector<learnStr> learns;

        public:
        Learner(void);
        virtual ~Learner(void);

            //  object management
            // =================================================================
	void add(core::LayerPtr src,core::LayerPtr dst,boost::python::numeric::array params);
	void learn(float scale = 1.0);
            // python export
            // =================================================================
            static void boost (void);
    };
}} // namespace dana::learn

#endif
