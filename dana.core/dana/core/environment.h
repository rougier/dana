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

#ifndef __DANA_CORE_ENVIRONMENT_H__
#define __DANA_CORE_ENVIRONMENT_H__

#include <boost/python.hpp>
#include <vector>
#include "object.h"
#include "map.h"

using namespace boost::python;


namespace dana { namespace core {

    typedef boost::shared_ptr<class Environment> EnvironmentPtr;

    class Environment : public Object {

        // ___________________________________________________________attributes
    public:
        std::vector<MapPtr>  maps;
        SpecPtr              spec;
        class Model *        model;
        
    public:
        // _________________________________________________________________life
        Environment (void);
        virtual ~Environment (void);

        // _________________________________________________________________main
        virtual void attach (MapPtr map);
        virtual void evaluate  (void);
        
        // ______________________________________________________________get/set
        virtual class Model * get_model (void);
        virtual void          set_model (class Model *model);
        virtual SpecPtr       get_spec (void);
        virtual void          set_spec (SpecPtr spec);  

        // _______________________________________________________________export
        static void python_export (void);
    };

}} // namespace dana::core

#endif
