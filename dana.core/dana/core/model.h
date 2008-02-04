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

#ifndef __DANA_CORE_MODEL_H__
#define __DANA_CORE_MODEL_H__

#include <vector>
#include "object.h"
#include "network.h"
#include "environment.h"
#include "observable.h"
#include "event.h"

namespace dana { namespace core {

typedef boost::shared_ptr<class Model> ModelPtr;

// _________________________________________________________________class Object
class Model : public Object, public Observable {

    // _______________________________________________________________attributes
public:
    std::vector<NetworkPtr>     networks;
    std::vector<EnvironmentPtr> environments;
    SpecPtr                     spec;
    unsigned long int           age;

        
public:
    // _____________________________________________________________________life
    Model (void);
    virtual ~Model (void);

    // _____________________________________________________________________main
    virtual void        append (NetworkPtr net);
    virtual void        append (EnvironmentPtr env);
    virtual void        compute_dp (void);
    virtual void        compute_dw (void);
    virtual void        evaluate (unsigned long n=1);
    virtual void        clear (void);
    
    // __________________________________________________________________get/set
    virtual unsigned long int get_age (void);
    virtual SpecPtr           get_spec (void);
    virtual void              set_spec (SpecPtr spec);  
        
    // ___________________________________________________________________export
    static void         python_export (void);
    };
}}

#endif
