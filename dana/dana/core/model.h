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
#include <vector>
#include "object.h"

using namespace boost::python;


namespace dana { namespace core {

    class Model : public Object {
        public:
            //  attributes
            // ================================================================
            std::vector<NetworkPtr>     networks;
            std::vector<EnvironmentPtr> environments;

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
            // =================================================================
            virtual void        evaluate (unsigned long n);

            // attribute manipulation
            // =================================================================

            // convenient methods
            // =================================================================

            // python export
            // =================================================================
            static void         boost (void);
    };

}} // namespace dana::core

#endif
