//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __DANA_CORE_MODEL_H__
#define __DANA_CORE_MODEL_H__

#include <boost/python.hpp>
#include <boost/thread/barrier.hpp>
#include <vector>
#include "object.h"

using namespace boost::python;


namespace dana { namespace core {

    class Model : public Object {
        public:
            //  attributes
            // ================================================================
            std::vector<NetworkPtr>  networks;
            std::vector<EnvironmentPtr> environments;
            bool running;
            boost::barrier *barrier;
            unsigned long age;
            unsigned long time, start_time, stop_time;
            static Model *current_model;
           
        public:
            // life management
            // =================================================================
            Model (void);
            virtual ~Model (void);

            // content management
            // =================================================================
            virtual void        append (NetworkPtr net);
            virtual void        append (EnvironmentPtr env);
            virtual void        clear (void);

            // activity management
            // ================================================================
            virtual bool        start (unsigned long n=0);
            virtual void        stop (void);
            static void         entry_point (void);

            // attribute manipulation
            // ================================================================

            // convenient methods
            // ================================================================

            // python export
            // =================================================================
            static void         boost (void);
    };

}} // namespace dana::core

#endif
