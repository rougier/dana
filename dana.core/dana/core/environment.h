//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: environment.h 275 2007-08-14 15:01:41Z rougier $

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
        public:
            std::vector<MapPtr>  maps;
            
        public:
            Environment (void);
            virtual ~Environment (void);
            void attach (MapPtr map);
            virtual void evaluate  (void);
            static void python_export (void);
    };

}} // namespace dana::core

#endif
