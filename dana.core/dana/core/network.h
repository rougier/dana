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

#ifndef __DANA_CORE_NETWORK_H__
#define __DANA_CORE_NETWORK_H__

#include <boost/python.hpp>
#include <vector>
#include "object.h"
#include "observable.h"
#include "map.h"


namespace dana { namespace core {

typedef boost::shared_ptr<class Network> NetworkPtr;

// ________________________________________________________________class Network
class Network : public Object, public Observable {
public:
    // _______________________________________________________________attributes
    std::vector<MapPtr>  maps; 
    unsigned int         width, height;
    unsigned long        age;
    SpecPtr              spec;
    class Model *        model;
    
public:
    // _____________________________________________________________________life
    Network (void);       
    virtual ~Network (void);
    
    // _____________________________________________________________________main
    virtual void         append (MapPtr layer);
    virtual MapPtr       get (const int index);
    virtual int          size (void) const;
    virtual void         compute_dp (void);
    virtual void         compute_dw (void);
    virtual void         clear (void);
    virtual void         compute_geometry (void);
    
    // ______________________________________________________________________I/O
    virtual int write (xmlTextWriterPtr writer);
    virtual int read  (xmlTextReaderPtr reader);
    
    // __________________________________________________________________get/set
    virtual class Model * get_model (void);
    virtual void          set_model (class Model *model);
    virtual SpecPtr       get_spec (void);
    virtual void          set_spec (SpecPtr spec);  
    virtual py::object    get_shape (void);
    
    // ___________________________________________________________________export        
    static void	        python_export (void);
};

}} // namespace dana::core
#endif
