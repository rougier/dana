//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: object.cc 241 2007-07-19 08:52:13Z rougier $

#include <ctime>
#include <unistd.h>
#include "object.h"
#include <libxml/xmlwriter.h>

using namespace dana::core;
using namespace boost;

// ________________________________________________________________runtime_error
void runtime_error (RuntimeError const &x)
{
    PyErr_SetString(PyExc_RuntimeError, x.what());
}

// _______________________________________________________________________Object
Object::Object (void)
{
    id = id_counter++;
}

// ______________________________________________________________________~Object
Object::~Object (void)
{}

// ___________________________________________________________________id_counter
unsigned int Object::id_counter = 1;

// _______________________________________________________________________myself
ObjectPtr
Object::myself (void) {
    if (_internal_weak_this.expired())
        throw RuntimeError("Shared pointer not available.");
    shared_ptr<Object> self = shared_from_this();
    assert(self != 0);
    return self;
}

// _________________________________________________________________________save
int
Object::save (const std::string filename)
{
    XMLNode main = XMLNode::createXMLTopNode("xml",TRUE);
    main.addAttribute("version","1.0");

    XMLNode node = main.addChild ("dana");

    //    std::ofstream file (filename.c_str());

    // Get time
    time_t rawtime;
    struct tm * timeinfo;
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    std::string date = asctime(timeinfo);
    date.erase (date.size()-1);   

    // Get author name
    std::string author = getlogin();

    // Get type
    std::string type = "archive";

    // Get version
    std::string version = "1.0";

    // Get comment
    std::string comment = "";

    node = node.addChild ("header");
    node.addAttribute ("version", version.c_str());
    node.addAttribute ("type", type.c_str());
    node.addAttribute ("date", date.c_str());
    node.addAttribute ("author", author.c_str());
    node.addAttribute ("comment", comment.c_str());
        
    
    main.writeToFile (filename.c_str());

    
    //    file << "DANA file" << std::endl;
    //    file << "version: " << version << std::endl;
    //     file << "type:    " << type    << std::endl;
    //     file << "date:    " << date    << std::endl;
    //     file << "author:  " << login   << std::endl;
    //     file << "comment: " << comment << std::endl;
    //     file << std::endl;

    //save (file);
    //    file.close();
    return 0;
}

// _________________________________________________________________________save
int
Object::save (std::ofstream &file)
{
    return 0;
}

// _________________________________________________________________________load
int
Object::load (const std::string filename)
{
    return 0;
}

// _________________________________________________________________________load
int
Object::load (std::ifstream &file)
{
    return 0;
}

// ________________________________________________________________python_export
void
Object::python_export (void)
{
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Object> >();
    register_exception_translator<RuntimeError>(&::runtime_error);

    int (Object::*save)(const std::string) = &Object::save;
    int (Object::*load)(const std::string) = &Object::load;

    class_<Object> ("Object", 
    "______________________________________________________________________\n"
    "                                                                      \n"
    "Based class for any object of dana.                                   \n"
    "                                                                      \n"
    "Attributes:                                                           \n"
    "   self -- Pointer on the underlying shared pointer if it exists.     \n"
    "______________________________________________________________________\n",
    
     init<>())
        .add_property ("self",
                       &Object::myself,
                       "underlying shared pointer (if it exists)")

        .def ("save", save,
              "save(filename) -> status")
        .def ("load", load,
              "load(filename) -> status")
        ;
}
