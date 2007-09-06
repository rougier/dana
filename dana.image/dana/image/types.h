//
// Copyright (C) 2006 Nicolas Rougier, Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
// 02111-1307, USA.
// $Id$

#ifndef __DANA_IMAGE_TYPES_H__
#define __DANA_IMAGE_TYPES_H__ 

typedef mirage::img::Coding<mirage::colorspace::GRAY_8>::Frame       ImageGray8;
typedef mirage::img::Coding<mirage::colorspace::RGB_24>::Frame       ImageRGB24;
typedef mirage::img::Coding<int>::Frame                              ImageInt;
typedef mirage::img::Coding<double>::Frame                           ImageDouble;
typedef mirage::img::Coding<mirage::colorspace::RGB<double> >::Frame ImageRGBDouble;

#endif
