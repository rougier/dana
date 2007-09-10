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

// _______________________________________________________________read_attribute
std::string
dana::core::read_attribute (xmlTextReaderPtr reader,
                            const char *name)
{
    xmlChar *tmp = xmlTextReaderGetAttribute (reader, BAD_CAST name);
    if (tmp != NULL) {
        std::string value = (char *) tmp;
        xmlFree (tmp);
        return value;
    }
    return std::string("");
}
    
// _______________________________________________________________________Object
Object::Object (void)
{
}

// ______________________________________________________________________~Object
Object::~Object (void)
{}

// ___________________________________________________________________id_counter
// unsigned long int Object::id_counter = 1;

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
               const std::string script_file,
               const std::string script_content)
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

    // Get version
    std::string version = "1.0";
    
    // Get comment
    std::string comment = "";
    
    // Actual save
    writer = xmlNewTextWriterFilename (uri.c_str(), 1);
    if (writer == NULL) {
        printf ("Error creating the xml writer\n");
        return 1;
    }
    xmlTextWriterSetIndent (writer, 1);
    xmlTextWriterSetIndentString (writer, BAD_CAST "    ");    
    xmlTextWriterStartDocument (writer, NULL, "utf-8", NULL);

    // <dana>
    xmlTextWriterStartElement (writer,
                               BAD_CAST "dana");
    xmlTextWriterWriteAttribute (writer,
                                 BAD_CAST "version",
                                 BAD_CAST version.c_str());
    xmlTextWriterWriteAttribute (writer,
                                 BAD_CAST "date",
                                 BAD_CAST date.c_str());
    xmlTextWriterWriteAttribute (writer,
                                 BAD_CAST "author",
                                 BAD_CAST author.c_str());

    // <script>
    xmlTextWriterStartElement (writer, BAD_CAST "script");
    xmlTextWriterWriteAttribute (writer,
                                 BAD_CAST "filename",
                                 BAD_CAST script_file.c_str());
    xmlTextWriterWriteString (writer,
                              BAD_CAST script_content.c_str());
    // </script>
    xmlTextWriterEndElement (writer);

    // <Object>
    write (writer);
    // </Object>

    // </dana>
    xmlTextWriterEndDocument(writer);
    
    xmlFreeTextWriter(writer);
    return 0;
}


// _________________________________________________________________________read
int
Object::read (const std::string filename)
{
    xmlTextReaderPtr reader;
    xmlReaderTypes   type   = XML_READER_TYPE_NONE;
    std::string      name   = "";
    int              status = 1;
    std::string version, date, author;
    
    reader = xmlReaderForFile (filename.c_str(), NULL, 1);
    if  (reader == NULL) {
        fprintf (stderr, "Unable to open %s\n", filename.c_str());
        return 1;
    }
        
    do  {
        status = xmlTextReaderRead (reader);
        if (status != 1)
            break;
        name = (char *) xmlTextReaderConstName(reader);
        type = (xmlReaderTypes) xmlTextReaderNodeType(reader);
        
        if (type == XML_READER_TYPE_ELEMENT) {
            if (name == "dana") {
                version = read_attribute (reader, "version");
                date    = read_attribute (reader, "date");
                author  = read_attribute (reader, "author");
                printf("Version: %s\n", version.c_str());
                printf("Date: %s\n",    date.c_str());
                printf("Author: %s\n",  author.c_str());
            }
            else if (name == "script") {
            }
            else {
                read (reader);
            }
        }
    } while (status == 1);

    xmlFreeTextReader(reader);
    if (status != 0) {
        fprintf (stderr, "%s : failed to parse\n", filename.c_str());
        return 1;
    }
    return 0;
}


// ________________________________________________________________python_export
void
Object::python_export (void)
{
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Object> >();
    register_exception_translator<RuntimeError>(&::runtime_error);

    int (Object::*write)(const std::string,
                         const std::string,
                         const std::string) = &Object::write;
    int (Object::*read)(const std::string) = &Object::read;

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
