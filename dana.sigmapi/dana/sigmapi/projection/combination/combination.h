//
// Copyright (C) 2006 Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#ifndef __DANA_SIGMAPI_PROJECTION_COMBINATION_H__
#define __DANA_SIGMAPI_PROJECTION_COMBINATION_H__

#include <boost/python.hpp>
#include <vector>
#include "core/link.h"
#include "core/layer.h"
#include "core/map.h"
#include "sigmapi/link.h"
//#include "core/object.h"

using namespace boost::python;


namespace dana { namespace sigmapi { namespace projection { namespace combination {

    // Forward declaration of shared pointers
    // =========================================================================
	typedef boost::shared_ptr<class Combination>  CombinationPtr;
	typedef boost::shared_ptr<class Linear> LinearPtr;
    
    // =========================================================================
	class Combination {
		public:
			Combination (void);
			virtual ~Combination ();
			virtual std::vector<core::LinkPtr> combine (core::UnitPtr dst,core::LayerPtr src1,core::LayerPtr src2);
	};
	
	class Linear : public Combination {
		private:
			float fac1_x;
			float fac2_x;
			float fac3_x;
			float fac1_y;
			float fac2_y;
			float fac3_y;
			float offset_x;
			float offset_y;
			float weight;
		public:
			Linear(float fac1_x,float fac2_x,float fac3_x,
			       float fac1_y,float fac2_y,float fac3_y,
			       float offset_x,float offset_y,float weight);
			std::vector<core::LinkPtr> combine (core::UnitPtr dst,core::LayerPtr src1,core::LayerPtr src2);
	};
}}}} // namespace dana::sigmapi::projection::combination

#endif
