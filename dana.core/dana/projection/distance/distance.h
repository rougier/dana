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
        

}}} // namespace dana::projection::distance

#endif


