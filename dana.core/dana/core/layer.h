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

#ifndef __DANA_CORE_LAYER_H__
#define __DANA_CORE_LAYER_H__

#include <vector>
#include "object.h"
#include "unit.h"
#include "link.h"
#include "spec.h"
#include "observable.h"


namespace dana { namespace core {

    typedef boost::shared_ptr<class Unit> UnitPtr;
    typedef boost::shared_ptr<class Layer> LayerPtr;

    // ______________________________________________________________class Layer
class Layer : public Object, public Observable {
    public:
        // ___________________________________________________________attributes
        class Map *          map; 
        std::vector<UnitPtr> units;
        std::vector<UnitPtr> permuted;
        SpecPtr              spec;
        py::object           potentials;

    public:
        // _________________________________________________________________life
        Layer (void);
        virtual ~Layer (void);

        // _________________________________________________________________main
        virtual void         append (UnitPtr unit);
        virtual UnitPtr      get (const int index) const;
        virtual UnitPtr      get (const int x, const int y) const;
        virtual int          size (void) const;
        virtual int          fill (py::object type);
        virtual void         clear (void);
        virtual float        compute_dp (void);
        virtual float        compute_dw (void);

        // __________________________________________________________________I/O
        virtual int write (xmlTextWriterPtr writer);
        virtual int read  (xmlTextReaderPtr reader);

        // ______________________________________________________________get/set
        virtual Map *        get_map (void);
        virtual void         set_map (class Map *map);
        virtual SpecPtr      get_spec (void) const;
        virtual void         set_spec (const SpecPtr spec);
        virtual py::object   get_shape (void);
        virtual py::object   get_potentials (void);
        virtual void         set_potentials (numeric::array a);

        // _______________________________________________________________export
        static void          python_export (void);
    };

}} // namespace dana::core

#endif
