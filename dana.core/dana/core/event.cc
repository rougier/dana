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


// _______________________________________________________________________export
void
Event::python_export (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Event> >();
    register_ptr_to_python< boost::shared_ptr<EventDP> >();
    register_ptr_to_python< boost::shared_ptr<EventDW> >();
    register_ptr_to_python< boost::shared_ptr<EventEvaluate> >();
 
    class_<Event, bases <Object> >(
        "Event",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__()"))
        ;

    class_<EventDP, bases <Event> >(
        "EventDP",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__()"))
        ;


    class_<EventDW, bases <Event> >(
        "EventDW",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__()"))
        ;

    class_<EventEvaluate, bases <Event> >(
        "EventEvaluate",
        "______________________________________________________________________\n"
        "                                                                      \n"
        "______________________________________________________________________\n",
        init<>("__init__()"))
        ;

}

