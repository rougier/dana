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
unsigned long int Object::id_counter = 1;

// _______________________________________________________________________myself
ObjectPtr
Object::myself (void) {
    if (_internal_weak_this.expired())
        throw RuntimeError("Shared pointer not available.");
    shared_ptr<Object> self = shared_from_this();
    assert(self != 0);
    return self;
}

// ________________________________________________________________________write
int
Object::write (xmlTextWriterPtr writer)
{
    return 0;
}

// _________________________________________________________________________read
int
Object::read (xmlTextReaderPtr reader)
{
    return 0;
}

// ________________________________________________________________________write
int
Object::write (const std::string filename,
               const std::string base,
               const std::string klass,
               const std::string module)
{
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
    xmlTextWriterWriteAttribute (writer, BAD_CAST "version",  BAD_CAST version.c_str());
    xmlTextWriterWriteAttribute (writer, BAD_CAST "type",  BAD_CAST type.c_str());
    xmlTextWriterWriteAttribute (writer, BAD_CAST "date",  BAD_CAST date.c_str());
    xmlTextWriterWriteAttribute (writer, BAD_CAST "author",  BAD_CAST author.c_str());

    // <Object>
    xmlTextWriterStartElement (writer, BAD_CAST base.c_str());
    xmlTextWriterWriteAttribute (writer, BAD_CAST "class",  BAD_CAST klass.c_str());
    xmlTextWriterWriteAttribute (writer, BAD_CAST "module", BAD_CAST module.c_str());
    xmlTextWriterWriteFormatAttribute (writer, BAD_CAST "id", "%d", id);

    write (writer);    

    // </Object>
    xmlTextWriterEndElement (writer);

    // </dana>
    xmlTextWriterEndDocument(writer);
    
    xmlFreeTextWriter(writer);
    return 0;
}

// ________________________________________________________________________write
void
Object::write_element (xmlTextWriterPtr writer,
                       std::string basetype,
                       ObjectPtr object)
{
    py::object obj (object->myself());
    py::object obj_class = obj.attr("__class__");
    std::string klass  = py::extract <std::string> (obj_class.attr("__name__"));
    std::string module = py::extract <std::string> (obj.attr("__module__"));

    xmlTextWriterStartElement (writer,
                               BAD_CAST basetype.c_str());
    xmlTextWriterWriteAttribute (writer,
                                 BAD_CAST "class",  BAD_CAST klass.c_str());
    xmlTextWriterWriteAttribute (writer,
                                 BAD_CAST "module", BAD_CAST module.c_str());
    xmlTextWriterWriteFormatAttribute (writer,
                                       BAD_CAST "id", "%d", object->id);
}

// _________________________________________________________________________load
int
Object::read (const std::string filename,
              const std::string base,
              const std::string klass,
              const std::string module)
{
    xmlTextReaderPtr reader;
    int ret;

    std::string version, date, author, type;    

    reader = xmlReaderForFile (filename.c_str(), NULL, 0);
    if (reader != NULL) {
        ret = xmlTextReaderRead (reader);

        while (ret == 1) {
            std::string name = (char *) xmlTextReaderConstName(reader);

            if ((xmlTextReaderNodeType(reader) == XML_ELEMENT_NODE) && (name == "dana")) {
                version = read_attribute (reader, "version");
                type    = read_attribute (reader, "type");
                date    = read_attribute (reader, "date");
                author  = read_attribute (reader, "author");
                printf("Type: %s\n",    type.c_str());
                printf("Version: %s\n", version.c_str());
                printf("Date: %s\n",    date.c_str());
                printf("Author: %s\n",  author.c_str());
            }
            
            else if ((xmlTextReaderNodeType(reader) == XML_ELEMENT_NODE)) {
                read (reader);
            }
            ret = xmlTextReaderRead(reader);
        }

        xmlFreeTextReader(reader);
        if (ret != 0) {
            fprintf(stderr, "%s : failed to parse\n", filename.c_str());
        }
    } else {
        fprintf(stderr, "Unable to open %s\n", filename.c_str());
    }
    return 0;
}

// _______________________________________________________________read_attribute
std::string
Object::read_attribute (xmlTextReaderPtr reader,
                        std::string name)
{
    xmlChar *tmp = xmlTextReaderGetAttribute (reader, BAD_CAST name.c_str());
    if (tmp != NULL) {
        std::string value = (char *) tmp;
        xmlFree (tmp);
        return value;
    }
    return std::string("");
}

// ________________________________________________________________python_export
void
Object::python_export (void)
{
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Object> >();
    register_exception_translator<RuntimeError>(&::runtime_error);

    int (Object::*write)(const std::string, const std::string,
                        const std::string, const std::string ) = &Object::write;
    int (Object::*read)(const std::string, const std::string,
                        const std::string, const std::string ) = &Object::read;

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

        .def ("write", write,
              "write(filename) -> status")
        .def ("read", read,
              "read(filename) -> status")
        ;
}
