//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __DANA_CORE_OBJECT_H__
#define __DANA_CORE_OBJECT_H__

#include <string>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>


namespace dana { namespace core {

    typedef boost::shared_ptr<class Object> ObjectPtr;
   
    struct RuntimeError {
        RuntimeError(std::string msg) : message(msg) { }
        const char *what() const throw() { return message.c_str(); }
        std::string message;
    };
    
    void runtime_error (RuntimeError const &x);

    class Object : public boost::enable_shared_from_this <Object> 
    {
        public:
            Object (void);
            virtual ~Object (void);
            virtual ObjectPtr self (void);
            static void python_export (void);
    };
}}

#endif
