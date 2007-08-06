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

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include "core/layer.h"
#include "core/object.h"
#include "../unit.h"
#include "../link.h"
#include "combination/combination.h"

using namespace boost::python;


namespace dana
{
namespace sigmapi
{
namespace projection
{

class Projection : public core::Object
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
    float dog(core::UnitPtr src,core::UnitPtr dst,float A,float a,float B,float b);

    static void static_connect (void);

public:
    core::LayerPtr          src1;
    core::LayerPtr          src2;
    core::LayerPtr		dst;
    bool                    self;

public:
    static void boost (void);
};

}
}
} // namespace dana::sigmapi::projection

#endif

