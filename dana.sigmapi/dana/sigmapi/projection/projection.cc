//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$

#include <iostream>
#include <math.h>
#include "projection.h"
#include "dana/core/map.h"
#include <boost/python/detail/api_placeholder.hpp>

using namespace dana::sigmapi::projection;

Projection *Projection::current = 0;

Projection::Projection (void) : Object ()
{
    self = true;
}

Projection::~Projection (void)
{}

void Projection::connect (void)
{
    std::vector<dana::core::LinkPtr> vlinkptr;
    for(int i = 0 ; i < dst->size() ; i++)
        {
            vlinkptr = combination->combine(dst->get(i),src1,src2);
            for(int j = 0 ; j < vlinkptr.size(); j++)
                ((dana::sigmapi::core::Unit*)(((dana::core::UnitPtr)(dst->get(i))).get()))->connect(vlinkptr[j]);
        }
}

void Projection::connect_all_to_one(float weight)
{
    dana::core::LinkPtr linkptr;
    dana::sigmapi::core::Link * link;
    linkptr = dana::core::LinkPtr (new dana::sigmapi::core::Link (dana::sigmapi::core::LinkType(dana::sigmapi::core::SIGMAPI_MAX)));
    link = (dana::sigmapi::core::Link*) (linkptr.get());
    for(int i = 0 ; i < src1->size();i++)
        {
            link->add_source(src1->get(i));
        }
    link->set_weight(weight);
    ((dana::sigmapi::core::Unit*)(((dana::core::UnitPtr)(dst->get(0))).get()))->connect(linkptr);
}

void Projection::connect_as_mod_cos(float scale_pos,float scale_neg)
{
	dana::core::LinkPtr linkptr;
	dana::sigmapi::core::Link * link;
    int width = src1->get_map()->width;
    float weight;
    if(src2 ->size()  == src1 -> size())
        for(int i = 0 ; i < dst->size();i++)
            {
                for(int j = 0 ; j < src1->size() ; j++)
                    {
                        linkptr = dana::core::LinkPtr (new dana::sigmapi::core::Link (dana::sigmapi::core::LinkType(dana::sigmapi::core::SIGMAPI_PROD)));
                        link = (dana::sigmapi::core::Link*) (linkptr.get());
                        link->add_source(src1->get(j));
                        link->add_source(src2->get(j));
                        weight = cos(2.0*M_PI*float(i-j)/width);
                        if(weight < 0)
                            weight = scale_neg * weight;
                        else
                            weight = scale_pos * weight;
                        link->set_weight(weight);
                        ((dana::sigmapi::core::Unit*)(((dana::core::UnitPtr)(dst->get(i))).get()))->connect(linkptr);
                    }
            }    
    else if(src2->size() == 1)
        {
            printf("mod cos , size(src2)  =1 Should not be used!!! \n");
            
            //         for(int i = 0 ; i < dst->size();i++)
            //             {
            //                 for(int j = 0 ; j < src1->size() ; j++)
            //                     {
            //                         linkptr = core::LinkPtr (new sigmapi::Link (LinkType(SIGMAPI_PROD)));
            //                         link = (dana::sigmapi::Link*) (linkptr.get());
            //                         link->add_source(src1->get(j));
            //                         link->add_source(src2->get(j));
            //                         weight = cos(2.0*M_PI*float(i-j)/width);
            //                         if(weight < 0)
            //                             weight = scale_neg * weight;
            //                         else
            //                             weight = scale_pos * weight;
            //                         link->set_weight(weight);
            //                         ((dana::sigmapi::Unit*)(((core::UnitPtr)(dst->get(i))).get()))->connect(linkptr);
            //                     }
            //             }   
        
        }
    
    else
        {
            std::cout << " Invalid map size " << std::endl;
        }
    
    
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
	dana::core::LinkPtr linkptr;
	dana::sigmapi::core::Link * link;

	for(int i = 0 ; i < dst->size();i++)
        {
            linkptr = dana::core::LinkPtr (new dana::sigmapi::core::Link (dana::sigmapi::core::LinkType(dana::sigmapi::core::SIGMAPI_MAX)));
            link = (dana::sigmapi::core::Link*) (linkptr.get());
            for(int j = 0 ; j < size ; j++)
                {
                    link->add_source(((dana::core::LayerPtr)(extract<dana::core::LayerPtr>(layers[j])))->get(i));
                }
            link->set_weight(weight);
            ((dana::sigmapi::core::Unit*)(((dana::core::UnitPtr)(dst->get(i))).get()))->connect(linkptr);		
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
            dana::core::LinkPtr linkptr;
            dana::sigmapi::core::Link * link;
            for(int i = 0 ; i < src1->size();i++)
                {
                    linkptr = dana::core::LinkPtr (new dana::sigmapi::core::Link (dana::sigmapi::core::LinkType(dana::sigmapi::core::SIGMAPI_PROD)));
                    link = (dana::sigmapi::core::Link*) (linkptr.get());
                    link->add_source(src1->get(i));
                    link->add_source(src2->get(0));
                    link->set_weight(weight);
                    ((dana::sigmapi::core::Unit*)(((dana::core::UnitPtr)(dst->get(i))).get()))->connect(linkptr);
                }
        }
    else
        {
            if(src2->size() != src1->size())
                {
                    printf("Cannot call connect_point_mod_one with size(src1)!=size(src2)");
                    return;
                }
            dana::core::LinkPtr linkptr;
            dana::sigmapi::core::Link * link;
            for(int i = 0 ; i < src1->size();i++)
                {
                    linkptr = dana::core::LinkPtr (new dana::sigmapi::core::Link (dana::sigmapi::core::LinkType(dana::sigmapi::core::SIGMAPI_PROD)));
                    link = (dana::sigmapi::core::Link*) (linkptr.get());
                    link->add_source(src1->get(i));
                    link->add_source(src2->get(i));
                    link->set_weight(weight);
                    ((dana::sigmapi::core::Unit*)(((dana::core::UnitPtr)(dst->get(i))).get()))->connect(linkptr);
                }
        }
}

void Projection::connect_dog_mod_one(float A,float a,float B,float b)
{
    dana::core::LinkPtr linkptr;
    dana::sigmapi::core::Link * link;
    if(src2->size() != 1 || src1->size() != dst->size())
        {
            printf("Cannot call connect_dog_mod_one with size(src1)!=size(dst) && size(src2)!=1");
            return;
        }
    for(int i = 0 ; i < dst->size();i++)
        {
            linkptr = dana::core::LinkPtr (new dana::sigmapi::core::Link (dana::sigmapi::core::LinkType(dana::sigmapi::core::SIGMAPI_PROD)));
            link = (dana::sigmapi::core::Link*) (linkptr.get());
            for(int j = 0 ; j < src1->size() ; j++)
                {
                    link->add_source(src1->get(j));
                    link->add_source(src2->get(0));
                    float weight = dog(src1->get(j),dst->get(i),A,a,B,b);
                    if(!(fabs(weight) <= 0.001f))
                        {
                            link->set_weight(weight);
                            ((dana::sigmapi::core::Unit*)(((dana::core::UnitPtr)(dst->get(i))).get()))->connect(linkptr);
                        }
                }
        }
}

float Projection::dog(dana::core::UnitPtr src,dana::core::UnitPtr dst,float A,float a,float B,float b)
{
    int dst_x = dst->get_x();
    int dst_y = dst->get_y();
    int src_x = src->get_x();
    int src_y = src->get_y();
    float distance = (src_x-dst_x)*(src_x-dst_x) + (src_y-dst_y)*(src_y-dst_y);
    return A * exp (-distance/(a*a)) - B * exp (-distance/(b*b));

}
