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

#ifndef __DANA_CORE_RANDOM_H__
#define __DANA_CORE_RANDOM_H__

#include <boost/python.hpp>
#include "object.h"


namespace dana { namespace core {

    class Random : public Object {
        public:
            //  attributes
            // ================================================================
            unsigned int seed;
           
        public:
            // life management
            // ================================================================
            Random (void);
            virtual ~Random (void);

            // content management
            // ================================================================
            void         set_seed (unsigned int seed);
            unsigned int get_seed (void);

            // python export
            // =================================================================
            static void  boost (void);
    };

}} // namespace dana::core

#endif
