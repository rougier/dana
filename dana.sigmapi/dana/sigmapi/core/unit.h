//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$

#ifndef __DANA_SIGMAPI_UNIT_H__
#define __DANA_SIGMAPI_UNIT_H__

#include <boost/python.hpp>
#include "dana/core/unit.h"
#include "link.h"

using namespace boost::python;

namespace dana
{
namespace sigmapi
{
namespace core
{
// Unit class
class Unit : public dana::core::Unit
{

public:
    // Input of the neuron : lateral and afferent contributions
    float input;
	
public:
    // Constructor
    Unit(void);

    // Connect
    void connect (dana::core::LinkPtr link);

    // Desctructor
    virtual ~Unit(void);

    // Evaluate new potential and return difference
    virtual float compute_dp (void);

    // Get the computed input of the neuron (lateral + afferent)
    float get_input(void) { return input;};
    //virtual int count_connections(void);
    
    // convenient methods
    // =================================================================
    virtual object      get_weights  (const dana::core::LayerPtr layer); 

public:
    // Boost python extension
    static void boost (void);
};

}
}
}// namespace dana::sigmapi::core

#endif
