//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id: model.h 143 2007-05-10 09:02:23Z rougier $

#ifndef __DANA_CORE_RANDOM_H__
#define __DANA_CORE_RANDOM_H__

#include <boost/python.hpp>
#include "object.h"


namespace dana { namespace core {

    class Random : public Object {
        public:
            //  attributes
            // ================================================================
            unsigned int seed;
           
        public:
            // life management
            // ================================================================
            Random (void);
            virtual ~Random (void);

            // content management
            // ================================================================
            void         set_seed (unsigned int seed);
            unsigned int get_seed (void);

            // python export
            // =================================================================
            static void  boost (void);
    };

}} // namespace dana::core

#endif
