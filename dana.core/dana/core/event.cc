//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <algorithm>
#include "event.h"
#include "observer.h"

using namespace dana::core;

// ________________________________________________________________________Event
Event::Event (ObjectPtr subject) : Object(), subject(subject)
{}

// _______________________________________________________________________~Event
Event::~Event (void)
{}

// _____________________________________________________________static observers
std::vector<ObserverPtr>  Event::observers;

// _______________________________________________________________________attach
void
Event::attach (ObserverPtr observer)
{
    std::vector<ObserverPtr>::iterator result;
    result = std::find (observers.begin(), observers.end(), observer);
    if (result != observers.end())
        return;    
    observers.push_back (ObserverPtr(observer));
}

// _______________________________________________________________________detach
void
Event::detach (ObserverPtr observer)
{
    std::vector<ObserverPtr>::iterator result;
    result = std::find (observers.begin(), observers.end(), observer);
    if (result != observers.end())
        observers.erase (result);
}

// _______________________________________________________________________notify
void
Event::notify (ObjectPtr subject)
{
    EventPtr event = EventPtr (new Event(subject));
    std::vector<ObserverPtr>::iterator i;
    for (i=observers.begin(); i!=observers.end(); i++)
        (*i)->notify (event);
}

// _____________________________________________________________static observers
std::vector<ObserverPtr>  EventDP::observers;

// _____________________________________________________________static observers
std::vector<ObserverPtr>  EventDW::observers;


// _______________________________________________________________________export
void
Event::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Event> >();
    register_ptr_to_python< boost::shared_ptr<EventDP> >();
    register_ptr_to_python< boost::shared_ptr<EventDW> >();
 
    class_<Event, bases <Object> >(
        "Event",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__(subject)"))

        .def ("attach",
              &Event::attach,
              "attach(observer)")
        .staticmethod ("attach")
        
        .def ("detach",
              &Event::detach,
              "detach(observer)")
        .staticmethod ("detach")
        ;

    class_<EventDP, bases <Event> >(
        "EventDP",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__(subject)"))
        ;


    class_<EventDW, bases <Event> >(
        "EventDW",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__(subject)"))
        ;

}

