/*
  DANA - Distributed Asynchronous Numerical Adaptive computing library
  Copyright (c) 2006,2007,2008 Nicolas P. Rougier

  This file is part of DANA.

  DANA is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free
  Software Foundation, either version 3 of the License, or (at your
  option) any later version.

  DANA is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  You should have received a copy of the GNU General Public License
  along with DANA. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __DANA_CORE_LINK_H__
#define __DANA_CORE_LINK_H__

#include "object.h"

namespace py = boost::python;

namespace dana { namespace core {

    typedef boost::shared_ptr<class Unit>  UnitPtr;
    typedef boost::shared_ptr<class Link>  LinkPtr;

    // _______________________________________________________________class Link
    class Link {
        // ___________________________________________________________attributes
    public:
        UnitPtr	          source;
        float	          weight;

    public:
        // _________________________________________________________________life
        Link (void);
        Link (UnitPtr source, float weight=0.0f);
        virtual ~Link (void);

        // _________________________________________________________________main
        virtual float              compute (void);

        // __________________________________________________________________I/O
        virtual int write (xmlTextWriterPtr writer);
        virtual int read  (xmlTextReaderPtr reader);

        // ______________________________________________________________get/set
        virtual py::tuple    const get (void);
        virtual UnitPtr      const get_source (void);
        virtual void	           set_source (UnitPtr src);
        virtual float        const get_weight (void);
        virtual void	           set_weight (float w);

        // _______________________________________________________________export
        static void                python_export (void);
    };

}}

#endif

