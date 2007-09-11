//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <cmath>
#include "vec4f.h"

using namespace glpython::objects;


Vec4f::Vec4f (float x, float y, float z, float w)
{
	this->x = x;
	this->y = y;
	this->z = z;
	this->w = w;
}

Vec4f::Vec4f (const Vec4f &other)
{
	x = other.x;
	y = other.y;
	z = other.z;
	w = other.w;
}

Vec4f::~Vec4f (void)
{}

void
Vec4f::zero (void)
{	
	x = y = z = w = 0;
}

void
Vec4f::normalize (void)
{
	float n = norm();
	if (n)
		scale (1.0f/n);
}

void
Vec4f::scale (const float value)
{
	x *= value;
	y *= value;
	z *= value;
	w *= value;
}

float
Vec4f::norm (void)
{
	float n = x*x + y*y + z*z + w*w;
    return sqrt(n);
}

float &
Vec4f::operator[] (unsigned int i)
{
	if (i == 0)
		return x;
	else if (i == 1)
		return y;
	else if (i == 2)
		return z;
	else
		return w;
}

Vec4f
Vec4f::cross (const Vec4f &other)
{
	return Vec4f (y*other.z-z*other.y,
                  z*other.x-x*other.z,
                  x*other.y-y*other.x,
                  0);
}

float
Vec4f::dot (const Vec4f &other)
{
	return x*other.x + y*other.y + z*other.z + w*other.w;
}

Vec4f
Vec4f::operator+ (const Vec4f &other)
{
	return Vec4f (x+other.x, y+other.y, z+other.z, w+other.w);
}


Vec4f
Vec4f::operator- (const Vec4f &other)
{
	return Vec4f (x-other.x, y-other.y, z-other.z, w-other.w);
}


Vec4f
Vec4f::operator* (const float value)
{
	return Vec4f (x*value, y*value, z*value, w*value);
}

int
Vec4f::operator== (const Vec4f &other)
{
	return x==other.x && y==other.y && z==other.z && w == other.w;
}

