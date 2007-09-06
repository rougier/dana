//
// Copyright (C) 2006,2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <iostream>
#include "projection.h"
#include "../core/map.h"
#include "../core/unit.h"
#include "../core/link.h"

using namespace dana::projection;


// ___________________________________________________________________Projection
Projection::Projection (core::LayerPtr src,
                        core::LayerPtr dst,
                        shape::ShapePtr shape,
                        profile::ProfilePtr profile,
                        density::DensityPtr density,
                        distance::DistancePtr distance,
                        bool self_connect): Object ()
{
    set_src (src);
    set_dst (dst);
    set_shape (shape);
    set_profile (profile);
    set_distance (distance);
    set_density (density);
    this->self_connect = self_connect;
}


// __________________________________________________________________~Projection
Projection::~Projection (void)
{}

// ______________________________________________________________________connect
void
Projection::connect (object data)
{
    int src_width  = src->map->width;
    int src_height = src->map->height;
    int dst_width  = dst->map->width;
    int dst_height = dst->map->height;

    // Relative shape from a unit
    std::vector<shape::vec2i> points;
    shape->call (points, distance, 0, 0, src_width, src_height);
    
    for (int i=0; i<dst->size(); i++) {
        int dst_x = (i % dst_width);
        int dst_y = (i / dst_width);
        float x0 = dst_x/float(dst_width);
        float y0 = dst_y/float(dst_height);
        //int src_x = int (dst_x * (float(src_width)/float(dst_width)));
        //int src_y = int (dst_y * (float(src_height)/float(dst_height)));
        //float cd = (*distance) (x0,y0,.5f,.5f);
        int src_x = int (dst_x * src_width/dst_width);
        int src_y = int (dst_y * src_height/dst_height);        
        for (unsigned int j=0; j<points.size(); j++) {
            int x = src_x - points[j].x;
            int y = src_y - points[j].y;
            if ((x >= 0) && (y>=0) && (x<src_width) && (y<src_height)) {
                float x1 = x/float(src_width);
                float y1 = y/float(src_height);
                float d = distance->call (x0,y0,x1,y1);
                float de = density->call (d);
                if ((de) && (self_connect || (dst->get(i) != src->get (y*src_width +x)))) {
                    float w = profile->call(d);
                    dst->get(i)->connect (src->get (y*src_width+x), w, data);
                }
            }
        }
    }    
}

// ____________________________________________________________________get_shape
shape::ShapePtr
Projection::get_shape (void)
{
    return shape::ShapePtr (shape);
}

// ____________________________________________________________________set_shape
void
Projection::set_shape (shape::ShapePtr shape)
{
    this->shape = shape::ShapePtr (shape);
}

// __________________________________________________________________get_profile
profile::ProfilePtr
Projection::get_profile (void)
{
    return profile::ProfilePtr(profile);
}

// __________________________________________________________________set_profile
void
Projection::set_profile (profile::ProfilePtr profile)
{
    this->profile = profile::ProfilePtr(profile);
}

// __________________________________________________________________get_density
density::DensityPtr
Projection::get_density (void)
{
    return density::DensityPtr (density);
}

// __________________________________________________________________set_density
void
Projection::set_density (density::DensityPtr density)
{
    this->density = density::DensityPtr (density);
}

// _________________________________________________________________get_distance
distance::DistancePtr
Projection::get_distance (void)
{
    return distance::DistancePtr(distance);
}

// _________________________________________________________________set_distance
void
Projection::set_distance (distance::DistancePtr distance)
{
    this->distance = distance::DistancePtr(distance);
}

// ______________________________________________________________________get_src
dana::core::LayerPtr
Projection::get_src (void)
{
    return core::LayerPtr(src);
}

// ______________________________________________________________________set_src
void
Projection::set_src (core::LayerPtr src)
{
    this->src = core::LayerPtr(src);
}

// ______________________________________________________________________get_dst
dana::core::LayerPtr
Projection::get_dst (void)
{
    return core::LayerPtr(dst);
}

// ______________________________________________________________________set_dst
void
Projection::set_dst (core::LayerPtr dst)
{
    this->dst = core::LayerPtr(dst);
}


