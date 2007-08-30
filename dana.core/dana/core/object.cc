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
#include <iostream>
#include "object.h"


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
Object::save (const std::string filename,
              const std::string base,
              const std::string klass,
              const std::string module)
{
//     py::object me(meme);
//     py::object klass = me.attr("__class__");
//     std::string klass_name = py::extract <std::string>  (klass.attr("__name__"));
//     std::string module_name = py::extract <std::string> (me.attr("__module__"));
//     std::string fullname = module_name + std::string (".") + klass_name;
//     std::cout << "Object type: " << fullname << std::endl;

    
    xmlTextWriterPtr writer;
    std::string uri = filename; //+ ".gz";

    // Collect information

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
    
    // Actual save
    writer = xmlNewTextWriterFilename (uri.c_str(), 0);
    if (writer == NULL) {
        printf ("Error creating the xml writer\n");
        return 1;
    }
    xmlTextWriterSetIndent (writer, 1);
    xmlTextWriterSetIndentString (writer, BAD_CAST "    ");    
    xmlTextWriterStartDocument (writer, NULL, "utf-8", NULL);

    // <dana>
    xmlTextWriterStartElement (writer, BAD_CAST "dana");

    // <meta>
    xmlTextWriterStartElement(writer, BAD_CAST "meta");

    xmlTextWriterWriteFormatElement (writer, BAD_CAST "version", "%s", BAD_CAST version.c_str());
    xmlTextWriterWriteFormatElement (writer, BAD_CAST "type",    "%s", BAD_CAST type.c_str());
    xmlTextWriterWriteFormatElement (writer, BAD_CAST "date",    "%s", BAD_CAST date.c_str());
    xmlTextWriterWriteFormatElement (writer, BAD_CAST "author",  "%s", BAD_CAST author.c_str());
    
    // </meta>
    xmlTextWriterEndElement(writer);

        
    // <Object>
    xmlTextWriterStartElement (writer, BAD_CAST base.c_str());
    xmlTextWriterWriteAttribute (writer, BAD_CAST "class",  BAD_CAST klass.c_str());
    xmlTextWriterWriteAttribute (writer, BAD_CAST "module", BAD_CAST module.c_str());

    save (writer);
    
    // </Object>
    xmlTextWriterEndElement (writer);

    // </dana>
    xmlTextWriterEndDocument(writer);
    
    xmlFreeTextWriter(writer);
    return 0;
}

// _________________________________________________________________________save
int
Object::save (xmlTextWriterPtr writer)
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

    int (Object::*save)(const std::string, const std::string,
                        const std::string, const std::string ) = &Object::save;
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
