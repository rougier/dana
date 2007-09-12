//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_OBSERVABLE_H__
#define __DANA_CORE_OBSERVABLE_H__

#include <vector>
#include "object.h"
#include "event.h"


namespace dana { namespace core {

typedef boost::shared_ptr<class Observable> ObservablePtr;
typedef boost::shared_ptr<class Observer> ObserverPtr;

// _____________________________________________________________class Observable
class Observable {

    // _______________________________________________________________attributes
public:
    std::vector<ObserverPtr>  observers;
        
public:
    // _____________________________________________________________________life
    Observable (void);
    virtual ~Observable (void);

    // _____________________________________________________________________main
    void attach (ObserverPtr observer, EventPtr event);
    void detach (ObserverPtr observer);
    void notify (EventPtr event);
        
    // ___________________________________________________________________export
    static void python_export (void);
};

}}

#endif
