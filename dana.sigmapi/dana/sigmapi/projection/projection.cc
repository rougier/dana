//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$

#include <iostream>
#include "projection.h"
#include "core/map.h"
#include <math.h>
#include <boost/python/detail/api_placeholder.hpp>

using namespace dana::sigmapi::projection;
//using namespace dana::sigmapi;

Projection *Projection::current = 0;

Projection::Projection (void) : Object ()
{
    self = true;
}

Projection::~Projection (void)
{}

void Projection::connect (void)
{
    std::vector<core::LinkPtr> vlinkptr;
    for(int i = 0 ; i < dst->size() ; i++)
    {
        vlinkptr = combination->combine(dst->get
                                        (i),src1,src2);
        for(int j = 0 ; j < vlinkptr.size(); j++)
            ((dana::sigmapi::Unit*)(((core::UnitPtr)(dst->get
                                     (i))).get()))->connect(vlinkptr[j]);
    }
}

void Projection::connect_all_to_one(float weight)
{
    core::LinkPtr linkptr;
    sigmapi::Link * link;
    linkptr = core::LinkPtr (new sigmapi::Link (LinkType(SIGMAPI_MAX)));
    link = (dana::sigmapi::Link*) (linkptr.get());
    for(int i = 0 ; i < src1->size();i++)
    {
        link->add_source(src1->get
                         (i));
    }
    link->set_weight(weight);
    ((dana::sigmapi::Unit*)(((core::UnitPtr)(dst->get
                             (0))).get()))->connect(linkptr);
}

void Projection::connect_max_one_to_one(boost::python::list layers, float weight)
{
	// The pattern of connections is one to one
	// The dst neuron compute a max over its inputs
	if(!PyList_Check(layers.ptr())){
		PyErr_SetString(PyExc_ValueError, "expected a PyArrayObject");
		throw_error_already_set();
	}

	int size = boost::python::len(layers);
	core::LinkPtr linkptr;
	sigmapi::Link * link;

	for(int i = 0 ; i < dst->size();i++)
	{
		linkptr = core::LinkPtr (new sigmapi::Link (LinkType(SIGMAPI_MAX)));
		link = (dana::sigmapi::Link*) (linkptr.get());
		for(int j = 0 ; j < size ; j++)
		{
			link->add_source(((core::LayerPtr)(extract<core::LayerPtr>(layers[j])))->get(i));
		}
		link->set_weight(weight);
		((dana::sigmapi::Unit*)(((core::UnitPtr)(dst->get
				(i))).get()))->connect(linkptr);		
	}
}

void Projection::connect_point_mod_one(float weight)
{
    if(src1->size()!=dst->size())
    {
        printf("Cannot call connect_point_mod_one with size(src1)!=size(dst)");
        return;
    }
    if(src2->size() == 1)
    {
        core::LinkPtr linkptr;
        sigmapi::Link * link;
        for(int i = 0 ; i < src1->size();i++)
        {
            linkptr = core::LinkPtr (new sigmapi::Link (LinkType(SIGMAPI_PROD)));
            link = (dana::sigmapi::Link*) (linkptr.get());
            link->add_source(src1->get
                             (i));
            link->add_source(src2->get
                             (0));
            link->set_weight(weight);
            ((dana::sigmapi::Unit*)(((core::UnitPtr)(dst->get
                                     (i))).get()))->connect(linkptr);
        }
    }
    else
    {
        if(src2->size() != src1->size())
        {
            printf("Cannot call connect_point_mod_one with size(src1)!=size(src2)");
            return;
        }
        core::LinkPtr linkptr;
        sigmapi::Link * link;
        for(int i = 0 ; i < src1->size();i++)
        {
            linkptr = core::LinkPtr (new sigmapi::Link (LinkType(SIGMAPI_PROD)));
            link = (dana::sigmapi::Link*) (linkptr.get());
            link->add_source(src1->get
                             (i));
            link->add_source(src2->get
                             (i));
            link->set_weight(weight);
            ((dana::sigmapi::Unit*)(((core::UnitPtr)(dst->get
                                     (i))).get()))->connect(linkptr);
        }
    }
}

void Projection::connect_dog_mod_one(float A,float a,float B,float b)
{
    core::LinkPtr linkptr;
    sigmapi::Link * link;
    if(src2->size() != 1 || src1->size() != dst->size())
    {
        printf("Cannot call connect_dog_mod_one with size(src1)!=size(dst) && size(src2)!=1");
        return;
    }
    for(int i = 0 ; i < dst->size();i++)
    {
        linkptr = core::LinkPtr (new sigmapi::Link (LinkType(SIGMAPI_PROD)));
        link = (dana::sigmapi::Link*) (linkptr.get());
        for(int j = 0 ; j < src1->size() ; j++)
        {
            link->add_source(src1->get
                             (j));
            link->add_source(src2->get
                             (0));
            float weight = dog(src1->get
                               (j),dst->get
                               (i),A,a,B,b);
            if(!(fabs(weight) <= 0.01))
            {
                link->set_weight(weight);
                ((dana::sigmapi::Unit*)(((core::UnitPtr)(dst->get
                                         (i))).get()))->connect(linkptr);
            }
        }
    }
}

float Projection::dog(core::UnitPtr src,core::UnitPtr dst,float A,float a,float B,float b)
{
    int dst_x = dst->get_x();
    int dst_y = dst->get_y();
    int src_x = src->get_x();
    int src_y = src->get_y();
    float distance = (src_x-dst_x)*(src_x-dst_x) + (src_y-dst_y)*(src_y-dst_y);
    return A * exp (-distance/(a*a)) - B * exp (-distance/(b*b));

}

// =============================================================================
//
// =============================================================================
void Projection::static_connect (void)
{
    if (current)
        current->connect();
}

// =============================================================================
//    Boost wrapping code
// =============================================================================
void Projection::boost (void)
{
    import_array(); // Important ! Sinon c'est seg fault
    class_<Projection>("projection",
                       "======================================================================\n"
                       "\n"
                       "A projection is the specification of a pattern of connection between\n"
                       "three layers. It can be precisely defined using :\n"
                       "\n"
                       "- a combination function : it defines how to combine the inputs to define\n"
                       "                          the links to each destination's neuron\n"
                       "\n"
                       "Attributes:\n"
                       "-----------\n"
                       " self:     whether self connections are to be made\n"
                       " src1:     First source layer\n"
                       " src2:     Second source layer\n"
                       " dst:      destination layer\n"
                       "\n"
                       "======================================================================\n",
                       init<> ("init() -- initializes the projection\n"))
    .def_readwrite ("self", &Projection::self)
    .def_readwrite ("src1", &Projection::src1)
    .def_readwrite ("src2", &Projection::src2)
    .def_readwrite ("dst", &Projection::dst)
    .def_readwrite ("combination", &Projection::combination)
    .def ("connect", &Projection::connect,
          "connect() -- instantiates the connection\n")
    .def("connect_all_to_one", &Projection::connect_all_to_one)
    .def("connect_max_one_to_one",&Projection::connect_max_one_to_one)
    .def("connect_point_mod_one", &Projection::connect_point_mod_one)
    .def("connect_dog_mod_one",&Projection::connect_dog_mod_one)
    ;
}

// ===================================================================
//  Boost module
// ===================================================================
BOOST_PYTHON_MODULE(_projection)
{
    using namespace boost::python;
    Projection::boost();
}
