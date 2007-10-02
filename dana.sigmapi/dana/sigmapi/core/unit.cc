//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id: unit.cc 262 2007-08-06 12:02:43Z fix $

#include "dana/core/map.h"
#include "dana/core/layer.h"
#include "dana/core/spec.h"
#include "dana/cnft/spec.h"
#include "link.h"
#include "unit.h"
#include <iostream>
#include <numpy/arrayobject.h>

using namespace boost::python::numeric;
using namespace dana::sigmapi::core;

// Constructor
// -----------------------------------------------------------------------------
Unit::Unit(void) : dana::core::Unit()
{
    input = 0;
}

// Destructor
// -----------------------------------------------------------------------------
Unit::~Unit(void)
{}

// Connect
// -----------------------------------------------------------------------------
void
Unit::connect(dana::core::LinkPtr link)
{
    afferents.push_back(link);
}



// Evaluate new potential and returns difference
// -----------------------------------------------------------------------------
float
Unit::compute_dp (void)
{
    dana::core::SpecPtr sp = layer->get_spec();
    if (sp.get() == NULL)
        return 0.0f;
    dana::cnft::Spec *s = dynamic_cast<cnft::Spec *>(sp.get());
    if (s == 0)
        return 0.0f;

    float tau      = s->tau;
    float alpha    = s->alpha;
    float baseline = s->baseline;
    float min_act  = s->min_act;
    float max_act  = s->max_act;    

    float input = 0;
    unsigned int size = afferents.size();

    for (unsigned int i=0; i<size; i++)
    {
        input += afferents[i]->compute();
    }

    float lateral = 0;
    size = laterals.size();

    for (unsigned int i=0; i<size; i++)
    {
        lateral += laterals[i]->compute();
    }
    this->input = input ; //(1.0f/alpha)*(lateral + input);
    float du = (-potential + baseline + (1.0f/alpha)*(lateral + input)) / tau;
    float value = potential;
    potential += du;

    if (potential < min_act)
        potential = min_act;

    if (potential > max_act)
        potential = max_act;

    return value-potential;
}

// ______________________________________________________________writeGraph
std::map<std::string, double>
Unit::writeGraph(std::ofstream& file)
{
    //printf("[sigmapi::core::Unit] Parse afferents \n");
    std::map<std::string, double> links;
    std::map<std::string, int> nb_links;
    std::string _name;
    std::map<std::string, double>::iterator iter1; // Iterator for the map links
    std::map<std::string, int>::iterator iter2; // Iterator for the map nb_links
    dana::core::UnitPtr unit;
    dana::core::LinkPtr link_tmp;
    dana::sigmapi::core::Link * sigpilat;
    dana::sigmapi::core::Link * sigpiaff;
    for (unsigned int i=0; i< laterals.size(); i++)
    {
        link_tmp = laterals[i];
        unit = link_tmp->get_source();
		if(unit == 0)
        {
            try
            {
                // The link is a sigmapi::Link
                // it is handled differently from the core::Link
                sigpilat = (dana::sigmapi::core::Link*)(link_tmp.get());
                for(unsigned int j = 0 ; j < sigpilat->sources.size() ; j++)
                {
                    boost::python::object o_map(sigpilat->get_source(j)->get_layer()->get_map()->myself());
                    _name = boost::python::extract <std::string> (o_map.attr("name"));                
                    iter1 = links.find(_name);
                    if(iter1 == links.end())
                    {
                        links[_name] = sigpilat->get_weight();
                        nb_links[_name] = 1;
                    }
                    else
                    {
                        // The map already exists in the map
                        links[_name] += sigpilat->get_weight();
                        nb_links[_name] += 1;
                    }
                }
            }
            catch(...)
            {
                std::cerr << "[ERROR] Unit::parse_afferents(lateral,sigpi) error" << std::endl;
                return links;
            }                
        }
        else
        {
            try
            {
                boost::python::object o_map(link_tmp->get_source()->get_layer()->get_map()->myself());
                _name = boost::python::extract <std::string> (o_map.attr("name"));
                iter1 = links.find(_name);
                if(iter1 == links.end())
                {
                    links[_name] = link_tmp->get_weight();
                    nb_links[_name] = 1;
                }
                else
                {
                    // The map already exists in the map
                    links[_name] += link_tmp->get_weight();
                    nb_links[_name] += 1;
                }
            }
            catch(...)
            {
                std::cerr << "[ERROR] Unit::parse_afferents(lateral,core) error" << std::endl;
                return links;
            }
        }
    }

    for (unsigned int i=0; i< afferents.size(); i++)
    {
        link_tmp = afferents[i];
        unit = link_tmp->get_source();
		if(unit == 0)
        {
            try
            {
                // The link is a sigmapi::Link
                // it is handled differently from the core::Link
                sigpiaff = (dana::sigmapi::core::Link*)(link_tmp.get());
                for(unsigned int j = 0 ; j < sigpiaff->sources.size() ; j++)
                {
                    boost::python::object o_map(sigpiaff->get_source(j)->get_layer()->get_map()->myself());
                    _name = boost::python::extract <std::string> (o_map.attr("name"));
                    iter1 = links.find(_name);
                    if(iter1 == links.end())
                    {
                        links[_name] = sigpiaff->get_weight();
                        nb_links[_name] = 1;
                    }
                    else
                    {
                        // The map already exists in the map
                        links[_name] += sigpiaff->get_weight();
                        nb_links[_name] += 1;
                    }                
                }
            }
            catch(...)
            {
                std::cerr << "[ERROR] Unit::parse_afferents(afferent,sigpi) error" << std::endl;
                return links;
            }
        }
        else
        {
            try
            {
                boost::python::object o_map(link_tmp->get_source()->get_layer()->get_map()->myself());
                _name = boost::python::extract <std::string> (o_map.attr("name"));          
                iter1 = links.find(_name);
                if(iter1 == links.end())
                {
                    links[_name] = link_tmp->get_weight();
                    nb_links[_name] = 1;
                }
                else
                {
                    // The map already exists in the map
                    links[_name] += link_tmp->get_weight();
                    nb_links[_name] += 1;
                }
            }
            catch(...)
            {
                std::cerr << "[ERROR] Unit::parse_afferents(afferent,core) error" << std::endl;
                return links;
            }
        }
    }

    // We finally compute the mean of the weights by dividing the weights
    // from each afferent unit by the total number of afferent units in this map
    for(iter1 = links.begin() ; iter1 != links.end() ; iter1++)
    {
        iter2 = nb_links.find((*iter1).first);
        (*iter1).second = (*iter1).second / double((*iter2).second);
    }
    return links;
}


object
Unit::get_weights (const dana::core::LayerPtr layer)
{
	if ((layer->map == 0) || (layer->map->width == 0)) {
		PyErr_SetString(PyExc_AssertionError, "layer has no shape");
		throw_error_already_set();
		return object();
	}

	unsigned int width = layer->map->width;
	unsigned int height = layer->map->height;
	
	npy_intp dims[2] = {height, width};
	object obj(handle<>(PyArray_SimpleNew (2, dims, PyArray_FLOAT)));

	PyArrayObject *array = (PyArrayObject *) obj.ptr();

	PyArray_FILLWBYTE(array, 0);

	float *data = (float *) array->data;
	const std::vector<dana::core::LinkPtr> *wts;
	if (layer.get() == this->layer) {
		wts = &laterals;
	} else {
		wts = &afferents;
	}

	for (unsigned int i=0; i< wts->size(); i++) {
		dana::core::UnitPtr unit = wts->at(i)->source;
		if(unit == 0)
		{
 			// It means that wts->at(i) is a sigmapi::Link
 			// in which the source neurons are managed in a different way
 			unit = ((dana::sigmapi::core::Link*)(wts->at(i).get()))->get_source(0);
 		}
 		if (unit->layer == layer.get())
 			if ((unit->y > -1) && (unit->x > -1))
 				data[unit->y*width+unit->x] += wts->at(i)->weight;
	}
	return extract<boost::python::numeric::array>(obj);	
}

/*int
  Unit::count_connections(void)
  {
  int numb = 0;
  for (unsigned int i=0; i<afferents.size(); i++)
  {
  numb += afferents[i]->count_connections();
  }
  for (unsigned int i=0; i<laterals.size(); i++)
  {
  numb += laterals[i]->count_connections();
  }
  return numb;
  }*/


// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Unit::boost (void)
{
    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Unit> >();
    import_array();
    class_<Unit, bases<dana::core::Unit> >("Unit",
                                           "======================================================================\n"
                                           "\n"
                                           "A unit is a potential that is computed on the basis of some external\n"
                                           "sources that feed the unit. Those sources can be anything as long as\n"
                                           "they have some potential.\n"
                                           "\n"
                                           "Attributes:\n"
                                           "-----------\n"
                                           "   potential : unit potential (float)\n"
                                           "   position  : unit position within layer (tuple, read only)\n"
                                           "\n"
                                           "======================================================================\n",
                                           init<>(
                                               "__init__ () -- initialize unit\n"))
        .def_readonly ("input", &Unit::get_input)
        ;
}
