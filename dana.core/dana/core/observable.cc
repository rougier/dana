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
#include "observable.h"
#include "event.h"

using namespace dana::core;

// ____________________________________________________________________Observable
Observable::Observable (void)
{}

// ___________________________________________________________________~Observable
Observable::~Observable (void)
{}

// _______________________________________________________________________attach
void
Observable::attach (ObserverPtr observer,
                    EventPtr event)
{
    std::vector<ObserverPtr>::iterator result;
    result = std::find (observers.begin(), observers.end(), observer);
    if (result != observers.end())
        return;    
    observers.push_back (ObserverPtr(observer));
    observer->event = EventPtr (event);
}

// _______________________________________________________________________detach
void
Observable::detach (ObserverPtr observer)
{
    std::vector<ObserverPtr>::iterator result;
    result = std::find (observers.begin(), observers.end(), observer);
    if (result != observers.end())
        observers.erase (result);
}

// _______________________________________________________________________notify
void
Observable::notify (EventPtr event)
{
    std::vector<ObserverPtr>::iterator i;
    for (i=observers.begin(); i!=observers.end(); i++)
        if (event->name == (*i)->event->name)
            (*i)->notify (event);
}

// _______________________________________________________________________export
void
Observable::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Observable> >();
 
    class_<Observable>(
        "Observable",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__()"))
        
        .def ("attach",
              &Observable::attach,
              "attach(observer)")
        
        .def ("detach",
              &Observable::detach,
              "detach(observer)")
        ;
}

