//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_EVENT_H__
#define __DANA_CORE_EVENT_H__

#include <vector>
#include "object.h"
#include "observer.h"


namespace dana { namespace core {

typedef boost::shared_ptr<class Event> EventPtr;
typedef boost::shared_ptr<class EventDP> EventDPPtr;
typedef boost::shared_ptr<class EventDW> EventDWPtr;

// __________________________________________________________________class Event
class Event : public Object {
public:
    std::string name;
    Event (std::string name = "Event") : Object(), name(name)  {};
    static void python_export (void);
};

// ________________________________________________________________class EventDP
class EventDP : public Event {
public:
    EventDP (void) : Event("EventDP") { };
    static void python_export (void);
};

// ________________________________________________________________class EventDW
class EventDW : public Event {
public:
    EventDW (void) : Event("EventDW") { };    
    static void python_export (void);
};

}}

#endif
