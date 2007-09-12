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
#include "sunit.h"
#include "dana/core/layer.h"
#include "dana/core/object.h"
#include <numpy/arrayobject.h>

using namespace boost::python;

namespace dana { namespace learn {

	typedef struct {
		core::LayerPtr source;
		core::LayerPtr destination;
		std::vector< std::vector<float> > params;
	} learnStr;
	
    class Learner {
	    private:
		    std::vector<learnStr> learns;
		    std::vector<std::vector<float> > learn_params;
		    core::LayerPtr src;
		    core::LayerPtr dst;

        public:
        Learner(void);
        virtual ~Learner(void);

	//  object management
	// =================================================================
	void set_source(core::LayerPtr src);
	void set_destination(core::LayerPtr dst);
	void add_one(boost::python::list params);
	void connect(void);
	
	void learn(float scale = 1.0);
	// python export
	// =================================================================
	static void boost (void);
    };
}} // namespace dana::learn

#endif
