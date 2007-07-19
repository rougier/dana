//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_SPEC_H__
#define __DANA_CORE_SPEC_H__

#include "object.h"

using namespace boost::python;

namespace dana { namespace core {

    typedef boost::shared_ptr<class Spec> SpecPtr;

    class Spec : public Object {
        public:
            Spec (void);
            virtual ~Spec (void);
            static void python_export (void);
    };
}}

#endif
