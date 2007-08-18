//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#ifndef __DANA_PROJECTION_PROJECTOR_H__
#define __DANA_PROJECTION_PROJECTOR_H__

#include <boost/python.hpp>
#include <vector>
#include "../core/object.h"
#include "projection.h"


using namespace boost::python;


namespace dana { namespace projection {

    // Forward declaration of shared pointers
    typedef boost::shared_ptr<class Projector>    ProjectorPtr;
    
    class Projector : public object
    {
        public:
            //  attributes
            // =================================================================
            std::vector<ProjectionPtr> projections;
            
        public:
            // life management 
            // =================================================================
            Projector (void);
            virtual ~Projector (void);

            // content management
            // =================================================================
            virtual void          append (ProjectionPtr layer);
            virtual ProjectionPtr get (const int index) const;
            virtual int           size (void) const;
            virtual void          clear (void);
            
            // activity management
            // =================================================================
            virtual void        connect  (bool use_thread);

        public:
            // python export
            // =================================================================
            static void         boost (void);
    };

}} // namespace dana::projection
#endif
