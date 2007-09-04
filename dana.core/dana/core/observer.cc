//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "event.h"
#include "observer.h"

using namespace dana::core;

// _____________________________________________________________________Observer
Observer::Observer (void) : Object()
{}

// ____________________________________________________________________~Observer
Observer::~Observer (void)
{}

// _______________________________________________________________________notify
void
Observer::notify (EventPtr event)
{}


// ______________________________________________________________ObserverWrapper

class ObserverWrapper:  public Observer, public py::wrapper<Observer> {
public:
    ObserverWrapper (void) : Observer() {};
    
    void notify (EventPtr event)
    {
        if (py::override notify = this->get_override("notify")) {
            notify (event);
            return;
        }
        Observer::notify(event);
    }
    void default_notify (EventPtr event)
    {
        this->Observer::notify(event);
    }
};

// _______________________________________________________________________export

void
Observer::python_export (void)
{
    using namespace boost::python;    
    register_ptr_to_python< boost::shared_ptr<Observer> >();
 
    class_<ObserverWrapper, bases <Object>, boost::noncopyable >(
        "Observer",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__()"))
        
        .def ("notify", &Observer::notify, &ObserverWrapper::default_notify,
              "notify(event)")
        ;
}

