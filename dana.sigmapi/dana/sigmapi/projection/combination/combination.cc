//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$

#include "combination.h"
#include <stdio.h>

using namespace dana;
using namespace dana::sigmapi::projection;
using namespace dana::sigmapi::projection::combination;


// =============================================================================
//
// =============================================================================

Combination::Combination()
{}

// =============================================================================
//
// =============================================================================

Combination::~Combination()
{}

// =============================================================================
//
// =============================================================================

std::vector<core::LinkPtr> Combination::combine (core::UnitPtr dst,core::LayerPtr src1,core::LayerPtr src2)
{
    std::vector<core::LinkPtr> links;
    return links;
}

// =============================================================================
//
// =============================================================================

Linear::Linear(float fac1_x,float fac2_x,float fac3_x,
               float fac1_y,float fac2_y,float fac3_y,
               float offset_x,float offset_y,float weight) : Combination()
{
    this->fac1_x = fac1_x;
    this->fac2_x = fac2_x;
    this->fac3_x = fac3_x;
    this->fac1_y = fac1_y;
    this->fac2_y = fac2_y;
    this->fac3_y = fac3_y;
    this->offset_x = offset_x;
    this->offset_y = offset_y;
    this->weight = weight;
}

std::vector<core::LinkPtr> Linear::combine (core::UnitPtr dst,core::LayerPtr src1,core::LayerPtr src2)
{
	// TODO : Inverser fac2 et fac3 (cf ligne suivante) , c'est illogique.
    // fac1 for destination, fac2 for input2 and fac3 for input1
    // fac1_x*i + offset_x = fac2_x*m + fac3_x*k
    // fac1_y*j + offset_y = fac2_y*n + fac3_y*l

    // La vraie formule est : fac2*src2 + fac3 * src1 = fac1 * dst + offset
    
    std::vector<core::LinkPtr> links;
    if(fac2_x != 0 && fac2_y != 0)
    {
        core::LinkPtr linkptr;
        sigmapi::Link * link;
        int dst_x = dst->get_x();
        int dst_y = dst->get_y();
        int src1_width  = src1->map->width;
        int src2_width = src2->map->width;
        int src2_height = src2->map->height;

        for(int i = 0 ; i < src1->size();i++)
        {
            int src1_x = (i % src1_width);
            int src1_y = (i / src1_width);
            if(fac2_x != 0 && fac2_y != 0)
            {
                int src2_x = int((fac1_x*dst_x+offset_x-fac3_x*src1_x)/fac2_x);
                int src2_y = int((fac1_y*dst_y+offset_y-fac3_y*src1_y)/fac2_y);
                if((src2_x>=0)&&
                        (src2_x<src2_width)&&
                        (src2_y>=0)&&
                        (src2_y<src2_height))
                {
                    linkptr = core::LinkPtr (new sigmapi::Link (sigmapi::LinkType(SIGMAPI_PROD)));
                    link = (dana::sigmapi::Link*) (linkptr.get());
                    link->add_source(src1->get(i));
                    link->add_source(src2->get(src2_x+src2_y*src2_width));
                    link->set_weight(weight);
                    links.push_back(linkptr);
                }
            }
        }
    }
    else
    {
        // En pratique, utilisée quand fac2x = fac2y = 0
        // Hypothèse supplémentaire : size(src2) = 1
        // Cette méthode est utilisée pour réaliser un facteur d'échelle entre 2 cartes
        // Par exemple : projeter une carte nxn sur une sous partie d'une autre carte nxn;
	if(src2->size() != 1)
	{
		printf("[Erreur] Cannot use Combine.linear with fac2 = 0 and size(src2) != 1");
	}
	else
	{
		core::LinkPtr linkptr;
		sigmapi::Link * link;
		int dst_x = dst->get_x();
		int dst_y = dst->get_y();
		int src1_width  = src1->map->width;
		for(int i = 0 ; i < src1->size();i++)
		{
			int src1_x = (i % src1_width);
			int src1_y = (i / src1_width);	
			if(int(fac1_x*dst_x + offset_x - fac3_x*src1_x) == 0
					&& int(fac1_y*dst_y + offset_y - fac3_y*src1_y) == 0)
			{
				linkptr = core::LinkPtr (new sigmapi::Link (sigmapi::LinkType(SIGMAPI_PROD)));
				link = (dana::sigmapi::Link*) (linkptr.get());
				link->add_source(src1->get(i));
				link->add_source(src2->get(0));
				link->set_weight(weight);
				links.push_back(linkptr);			
			}
		}
	}
    }
    return links;
}

// =============================================================================
//    Boost wrapping code
// =============================================================================
BOOST_PYTHON_MODULE(_combination)
{
    using namespace boost::python;

    register_ptr_to_python< boost::shared_ptr<Combination> >();
    register_ptr_to_python< boost::shared_ptr<Linear> >();

    class_<Combination>("combination",
                        "======================================================================\n"
                        "\n"
                        " A Combination function determines how to combine the input layers    \n"
                        " to determine the links for a destination neuron                      \n"
                        "======================================================================\n",
                        init< > ("__init__() -- initializes profile\n"));

    class_<Linear, bases <Combination> >("linear",
                                         "======================================================================\n"
                                         "\n"
                                         " A Linear combinaton function combines linearly the two input layers  \n"
                                         "======================================================================\n",
                                         init<float,float,float,float,float,float,float,float,float>
                                         ("__init__(fac1_x,fac2_x,fac3_x,fac1_y,fac2_y,fac3_y,offset_x,offset_y,weight) -- initializes a linear combination\n"
                                          " faci_x are the factors of the linear combination along the x axe    \n"
                                          " faci_y are the factors of the linear combination along the y axe    \n"
                                          " The layers are combined as follows :\n"
                                          "    fac1_x*dst_x + offset_x = fac2_x*src1_x + fac3_x*src2_x \n"
                                          "    fac1_y*dst_y + offset_y = fac2_y*src1_y + fac3_y*src2_y \n"
                                         ));
}

