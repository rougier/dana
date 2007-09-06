//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_OBSERVER_H__
#define __DANA_CORE_OBSERVER_H__

#include "object.h"

namespace dana { namespace core {

    typedef boost::shared_ptr<class Event> EventPtr;
    typedef boost::shared_ptr<class Observer> ObserverPtr;
    
    // ___________________________________________________________class Observer
    class Observer : public Object {

        // ___________________________________________________________attributes
    public:
        
    public:
        // _________________________________________________________________life
        Observer (void);
        virtual ~Observer (void);

        // _________________________________________________________________main
        virtual void notify (EventPtr event);
        
        // _______________________________________________________________export
        static void  python_export (void);
    };
}}

#endif
