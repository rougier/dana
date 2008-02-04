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

#ifndef __DANA_CORE_EVENT_H__
#define __DANA_CORE_EVENT_H__

#include <vector>
#include "object.h"
#include "observer.h"


namespace dana { namespace core {

typedef boost::shared_ptr<class Event> EventPtr;
typedef boost::shared_ptr<class EventDP> EventDPPtr;
typedef boost::shared_ptr<class EventDW> EventDWPtr;
typedef boost::shared_ptr<class EventEvaluate> EventEvaluatePtr;

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

// ________________________________________________________________class EventDW
class EventEvaluate : public Event {
public:
    EventEvaluate (void) : Event("EventEvaluate") { };    
    static void python_export (void);
};

}}

#endif
