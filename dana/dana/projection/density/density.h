//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_PROJECTION_DENSITY_H__
#define __DANA_PROJECTION_DENSITY_H__

#include <boost/python.hpp>

using namespace boost::python;


namespace dana { namespace projection { namespace density {

    // Forward declaration of shared pointers
    // =========================================================================
    typedef boost::shared_ptr<class Density> DensityPtr;
    typedef boost::shared_ptr<class Full>    FullPtr;
    typedef boost::shared_ptr<class Sparse>  SparsePtr;
    typedef boost::shared_ptr<class Sparser> SparserPtr;            

    // =========================================================================
    class Density {
        public:
            float density;
        public:
            Density (float d = 1.0f);
            virtual ~Density();
            virtual bool call (float distance);          
    };
    
    // =========================================================================
    class Full : public Density {
        public:
            Full (float d = 1.0f);
            bool call (float distance);
    };

    // =========================================================================
    class Sparse : public Density {
        public:
            Sparse (float d = 1.0f);
            bool call (float distance);
    };

    // =========================================================================
    class Sparser : public Density {
        public:
            Sparser (float d = 1.0f);
            bool call (float distance);
    };

}}} // namespace dana::projection::density

#endif


