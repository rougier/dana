/*
  DANA - Distributed Asynchronous Numerical Adaptive computing library
  Copyright (c) 2006,2007,2008 Nicolas P. Rougier

  This file is part of DANA.

  DANA is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free
  Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  DANA is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  You should have received a copy of the GNU General Public License
  along with DANA. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __DANA_CORE_SPEC_H__
#define __DANA_CORE_SPEC_H__

#include "object.h"

//using namespace boost::python;

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
