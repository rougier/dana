//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.



#ifndef __DANA_PROJECTION_DISTANCE_H__
#define __DANA_PROJECTION_DISTANCE_H__

#include <boost/python.hpp>

using namespace boost::python;


namespace dana { namespace projection { namespace distance {

    // Forward declaration of shared pointers
    // =========================================================================
    typedef boost::shared_ptr<class Distance>  DistancePtr;
    typedef boost::shared_ptr<class Euclidean> EuclideanPtr;
    typedef boost::shared_ptr<class Manhattan> ManhattanPtr;
    typedef boost::shared_ptr<class Max>       MaxPtr;   


    // =========================================================================
    class Distance {
        public:
            bool is_toric;
        public:
            Distance (bool toric=false);
            virtual ~Distance (void);
            virtual float call (float x0, float y0, float x1, float y1);
    };

    // =========================================================================
    class Euclidean : public Distance {
        public:
            Euclidean (bool toric=false);
            float call (float x0, float y0, float x1, float y1);
    };

    // =========================================================================
    class Manhattan : public Distance {
        public:
            Manhattan (bool toric=false);
            float call (float x0, float y0, float x1, float y1);
    };
    
    // =========================================================================
    class Max : public Distance {
        public:
            Max (bool toric=false);
            float call (float x0, float y0, float x1, float y1);
    };
        
    


    struct euclidean : public object {
        bool _toric;
        euclidean (bool toric) : object(), _toric(toric) {};
        float operator()(float x0, float y0, float x1, float y1)
        {
            float dx = fabs(x0-x1);
            float dy = fabs(y0-y1);
            if (_toric) {
                dx = ((1-dx) < dx) ? (1-dx) : dx;
                dy = ((1-dy) < dy) ? (1-dy) : dy;
            }
            return sqrt (dx*dx+dy*dy); 
        };
    };

    struct manhattan : public object {
        bool _toric;
        manhattan (bool toric) : object(), _toric(toric) {};
        float operator()(float x0, float y0, float x1, float y1)
        {
            float dx = fabs(x0-x1);
            float dy = fabs(y0-y1);
            if (_toric) {
                dx = ((1-dx) < dx) ? (1-dx) : dx;
                dy = ((1-dy) < dy) ? (1-dy) : dy;
            }
            return dx+dy;
        };
    };
    
     struct max : public object {
        bool _toric;
        max (bool toric) : object(), _toric(toric) {};
        float operator()(float x0, float y0, float x1, float y1)
        {
            float dx = fabs(x0-x1);
            float dy = fabs(y0-y1);
            if (_toric) {
                dx = ((1-dx) < dx) ? (1-dx) : dx;
                dy = ((1-dy) < dy) ? (1-dy) : dy;
            }
            return (dx > dy) ? dx : dy;
        };
    };

} // namespace distance
}} // namespace dana::projection

#endif


