//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "model.h"

using namespace boost::python;
using namespace glpython::objects;


//_______________________________________________________________________Model
Model::Model (std::string filename,
              float alpha,
              std::string name) : core::Object (name)

{
}


//_______________________________________________________________________~Model
Model::~Model (void)
{
}

//___________________________________________________________________initialize
void
Model::initialize (void)
{
}

//_______________________________________________________________________render
void
Model::render (void)
{}

//_______________________________________________________________________update
void
Model::update (void)
{}

//________________________________________________________________python_export
void
Model::python_export (void)
{

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Model> >();

    class_<Model, bases< core::Object> > ("Model",
    " ______________________________________________________________________\n"
    "                                                                       \n"
    " ______________________________________________________________________\n",
    init<std::string, optional <float, std::string> > (
        (arg("filename"),
         arg("alpha") = 1,
         arg("name") = "Model"),
        "__init__ (filename, alpha, name )\n"))
    ;
}
