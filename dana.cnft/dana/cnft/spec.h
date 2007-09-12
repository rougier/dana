//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_CNFT_SPEC_H__
#define __DANA_CNFT_SPEC_H__

#include <boost/python.hpp>
#include "dana/core/spec.h"

using namespace boost::python;

namespace dana { namespace cnft {

    class Spec : public core::Spec
    {
    	public:
    	    float tau;
            float alpha;
            float min_du;
            float baseline;
            float lrate;
            float min_act;
            float max_act;
            
        public:
            // Constructor
            Spec (void);

            // Destructor
            virtual ~Spec (void);

        public:
            // Boost python extension
            static void boost (void);
    };

}} // namespace dana::core

#endif
