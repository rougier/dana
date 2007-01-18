//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_NETWORK_H__
#define __DANA_CORE_NETWORK_H__

#include <boost/python.hpp>
#include <boost/thread/barrier.hpp>
#include <vector>
#include "object.h"
#include "map.h"

using namespace boost::python;


namespace dana { namespace core {

	typedef boost::shared_ptr<class Network> NetworkPtr;

    class Network : public Object {
	    public:
	        //  attributes
            // =================================================================
	        std::vector<MapPtr>  maps;         // maps composing the network
            unsigned int         width, height;// global shape
            unsigned long        age;          // age
            boost::barrier *     barrier;      // thread synchronization barrier
            object               spec;         // Specification of the network
            
	public:
            // life management 
            // =================================================================
            Network (void);       
            virtual ~Network (void);
            
            // content management
            // =================================================================
            virtual void         append (MapPtr layer);
            virtual MapPtr       get (const int index);
            virtual int          size (void) const;

            //  activity management
            // =================================================================
            virtual void         evaluate (unsigned long n=1,
                                           bool use_thread=false);
            virtual void         clear (void);
        
            //  attribute manipulation
            // =================================================================
            virtual object      get_shape (void);


            //  convenience functions
            // =================================================================
            virtual void        compute_geometry (void);

        public:
            // python export
            // =================================================================
            static void	        boost (void);
    };
    
}} // namespace dana::core
#endif
