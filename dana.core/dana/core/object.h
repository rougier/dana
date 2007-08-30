//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: object.h 241 2007-07-19 08:52:13Z rougier $


#ifndef __DANA_CORE_OBJECT_H__
#define __DANA_CORE_OBJECT_H__

#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <libxml/xmlwriter.h>


namespace py      = boost::python;
namespace numeric = boost::python::numeric;

namespace dana { namespace core {

    typedef boost::shared_ptr<class Object> ObjectPtr;
   
    struct RuntimeError {
        RuntimeError(std::string msg) : message(msg) { }
        const char *what() const throw() { return message.c_str(); }
        std::string message;
    };
    
    void runtime_error (RuntimeError const &x);

    // ___________________________________________________________________Object
    class Object : public boost::enable_shared_from_this <Object> {
    public:
        static unsigned int id_counter;
        unsigned int id;
        
    public:
        // _________________________________________________________________life
        Object (void);
        virtual ~Object (void);
        virtual ObjectPtr myself (void);

        // __________________________________________________________________I/O
        virtual int save (const std::string filename,
                          const std::string base,
                          const std::string klass,
                          const std::string module);
        virtual int save (xmlTextWriterPtr writer);
        virtual int load (const std::string filename);
        virtual int load (std::ifstream &file);

        // _______________________________________________________________export
        static void python_export (void);
    };
}}

#endif
