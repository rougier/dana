//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#ifndef __DANA_PROJECTION_PROFILE_H__
#define __DANA_PROJECTION_PROFILE_H__

#include <boost/python.hpp>

using namespace boost::python;


namespace dana { namespace projection { namespace profile {

    // Forward declaration of shared pointers
    // =========================================================================
    typedef boost::shared_ptr<class Profile>  ProfilePtr;
    typedef boost::shared_ptr<class Constant> ConstantPtr;
    typedef boost::shared_ptr<class Linear>   LinearPtr;
    typedef boost::shared_ptr<class Uniform>  UniformPtr;
    typedef boost::shared_ptr<class Gaussian> GaussianPtr;
    typedef boost::shared_ptr<class DoG>      DoGPtr;    

    // =========================================================================
    class Profile {
        public:
            float density;
        public:
            Profile (void);
            virtual ~Profile ();
            virtual float call (float distance);
    };
    
    // =========================================================================
    class Constant : public Profile {
        public:
            float value;
        public:
            Constant (float v);
            float call (float distance);
    };

    // =========================================================================
    class Linear : public Profile {
        public:
            float minimum, maximum;
        public:
            Linear (float min, float max);
            float call (float distance);
    };

    // =========================================================================
    class Uniform : public Profile {
        public:
            float minimum, maximum;
        public:
            Uniform (float min, float max);
            float call (float distance);
    };
    
    // =========================================================================
    class Gaussian : public Profile {
        public:
            float scale, mean;
        public:
            Gaussian (float s, float m);
            float call (float distance);
    };

    // =========================================================================
    class DoG : public Profile {
        public:
            float scale_1, mean_1;
            float scale_2, mean_2;            
        public:
            DoG (float s1, float m1, float s2, float m2);
            float call (float distance);
    };    


}}} // namespace dana::projection::density

#endif

