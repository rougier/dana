//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


/*
 * Object is the base class for all objects of DANA but currently offers
 * no specific method or attribute. It may change in the future.
 */
 

#ifndef __DANA_CORE_OBJECT_H__
#define __DANA_CORE_OBJECT_H__

#include <boost/python.hpp>

using namespace boost::python;


namespace dana { namespace core {

    // Forward declaration of shared pointers
    typedef boost::shared_ptr<class Model>          ModelPtr;
    typedef boost::shared_ptr<class Environment>    EnvironmentPtr;
    typedef boost::shared_ptr<class Network>        NetworkPtr; 
    typedef boost::shared_ptr<class Map>            MapPtr;
    typedef boost::shared_ptr<class Layer>          LayerPtr;
    typedef boost::shared_ptr<class Unit>           UnitPtr;
    typedef boost::shared_ptr<class Link>           LinkPtr;
    typedef boost::shared_ptr<class Spec>           SpecPtr;
    
    
    class Object /*: public object */ {
        public:
            Object (void);
            virtual ~Object (void);
    };

}} // namespace dana::core

#endif
