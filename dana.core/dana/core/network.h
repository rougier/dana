//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: network.h 138 2007-03-14 13:09:47Z rougier $

#ifndef __DANA_CORE_NETWORK_H__
#define __DANA_CORE_NETWORK_H__

#include <boost/python.hpp>
#include <vector>
#include "object.h"
#include "map.h"


namespace dana { namespace core {

	typedef boost::shared_ptr<class Network> NetworkPtr;

    // ____________________________________________________________class Network
    class Network : public Object {
    public:
        // ___________________________________________________________attributes
        std::vector<MapPtr>  maps; 
        unsigned int         width, height;
        unsigned long        age;
        SpecPtr              spec;
        class Model *        model;
            
	public:
        // _________________________________________________________________life
        Network (void);       
        virtual ~Network (void);

        // _________________________________________________________________main
        virtual void         append (MapPtr layer);
        virtual MapPtr       get (const int index);
        virtual int          size (void) const;
        virtual void         compute_dp (void);
        virtual void         compute_dw (void);
        virtual void         clear (void);
        virtual void         compute_geometry (void);
        
        // __________________________________________________________________I/O
        virtual int write (xmlTextWriterPtr writer);
        virtual int read  (xmlTextReaderPtr reader);

        // ______________________________________________________________get/set
        virtual class Model * get_model (void);
        virtual void          set_model (class Model *model);
        virtual SpecPtr       get_spec (void);
        virtual void          set_spec (SpecPtr spec);  
        virtual py::object    get_shape (void);

        // _______________________________________________________________export        
        static void	        python_export (void);
    };
    
}} // namespace dana::core
#endif
