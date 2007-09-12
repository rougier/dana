//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
// $Id$


#ifndef __DANA_SIGMAPI_PROJECTION_H__
#define __DANA_SIGMAPI_PROJECTION_H__


#include "dana/core/layer.h"
#include "dana/core/object.h"
#include "../core/unit.h"
#include "../core/link.h"
#include "combination/combination.h"

#include <boost/python.hpp>
#include <numpy/arrayobject.h>

//using namespace boost::python;


namespace dana
{
namespace sigmapi
{
namespace projection
{

class Projection : public dana::core::Object
{
public:
    static Projection *current;
    combination::CombinationPtr combination;

public:
    Projection (void);
    virtual ~Projection (void);
    void connect (void);
    void connect_all_to_one(float weight);
    void connect_as_mod_cos(float scale_pos,float scale_neg);
    
    void connect_max_one_to_one(boost::python::list layers, float weight);
    void connect_point_mod_one(float weight);
    void connect_dog_mod_one(float A,float a,float B,float b);
    float dog(dana::core::UnitPtr src,dana::core::UnitPtr dst,float A,float a,float B,float b);

    static void static_connect (void);

public:
    dana::core::LayerPtr          src1;
    dana::core::LayerPtr          src2;
    dana::core::LayerPtr		dst;
    bool                    self;
};

}
}
} // namespace dana::sigmapi::projection

#endif

