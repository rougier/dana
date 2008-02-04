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
#include <libxml/xmlreader.h>


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
    
    std::string read_attribute   (xmlTextReaderPtr reader,
                                  const char *name);

    // ___________________________________________________________________Object
    class Object : public boost::enable_shared_from_this <Object> {
    public:
        // _________________________________________________________________life
        Object (void);
        virtual ~Object (void);
        virtual ObjectPtr myself (void);

        // __________________________________________________________________I/O
        virtual int         write (xmlTextWriterPtr writer);
        virtual int         read (xmlTextReaderPtr reader);
        virtual int         write (const std::string filename,
                                   const std::string script_file ="None",
                                   const std::string script_content ="");
        virtual int         read  (const std::string filename);

        // _______________________________________________________________export
        static void python_export (void);
    };
}}

#endif
