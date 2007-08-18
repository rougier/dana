//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.



#ifndef __DANA_PROJECTION_SHAPE_H__
#define __DANA_PROJECTION_SHAPE_H__

#include <vector>
#include <cmath>
#include <boost/python.hpp>
#include "../distance/distance.h"

using namespace boost::python;
using namespace dana::projection;

namespace dana { namespace projection { namespace shape {

    // Definition of a coupe of integer
    typedef struct vec2i {
        int x, y;
        vec2i (int x_, int y_) : x(x_), y(y_) {};
    };

    // Forward declaration of shared pointers
    // =========================================================================
    typedef boost::shared_ptr<class Shape> ShapePtr;
    typedef boost::shared_ptr<class Point> PointPtr;
    typedef boost::shared_ptr<class Box>   BoxPtr;
    typedef boost::shared_ptr<class Disc>  DiscPtr;

    // =========================================================================
    class Shape {
        public:
            Shape (void);
            virtual ~Shape ();
            virtual int call (std::vector<vec2i> &points,
                              distance::DistancePtr distance,
                              int x, int y, int w, int h);
    };
    
    // =========================================================================
    class Point : public Shape {
        public:
            Point (void);
            int call (std::vector<vec2i> &points,
                      distance::DistancePtr distance,
                      int x, int y, int w, int h);
    };
    
    // =========================================================================
    class Box : public Shape {
        public:
            float width, height;

        public:
            Box (float w, float h);
            int call (std::vector<vec2i> &points,
                      distance::DistancePtr distance,
                      int x, int y, int w, int h);
    };   

    // =========================================================================
    class Disc : public Shape  {
        public:
            float radius;

        public:
            Disc (float r);
            int call (std::vector<vec2i> &points,
                      distance::DistancePtr distance,
                      int x, int y, int w, int h);
    };

}}} // namespace dana::projection::shape

#endif
