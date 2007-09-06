//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <cmath>
#include "distance.h"

using namespace dana::projection;
using namespace dana::projection::distance;


// =============================================================================
//
// =============================================================================
Distance::Distance (bool toric) : is_toric(toric)
{}

// =============================================================================
//
// =============================================================================
Distance::~Distance (void)
{}

// =============================================================================
//
// =============================================================================
float
Distance::call (float x0, float y0, float x1, float y1)
{
    return 0.0;
}

// =============================================================================
//
// =============================================================================
Euclidean::Euclidean (bool toric) : Distance(toric)
{}

// =============================================================================
//
// =============================================================================
float
Euclidean::call (float x0, float y0, float x1, float y1)
{
    float dx = fabs(x0-x1);
    float dy = fabs(y0-y1);
    if (is_toric) {
        dx = ((1-dx) < dx) ? (1-dx) : dx;
        dy = ((1-dy) < dy) ? (1-dy) : dy;
    }
    return sqrt (dx*dx+dy*dy); 
}

// =============================================================================
//
// =============================================================================
Manhattan::Manhattan (bool toric) : Distance(toric)
{}

// =============================================================================
//
// =============================================================================
float
Manhattan::call (float x0, float y0, float x1, float y1)
{
    float dx = fabs(x0-x1);
    float dy = fabs(y0-y1);
    if (is_toric) {
        dx = ((1-dx) < dx) ? (1-dx) : dx;
        dy = ((1-dy) < dy) ? (1-dy) : dy;
    }
    return dx+dy;
}

// =============================================================================
//
// =============================================================================
Max::Max (bool toric) : Distance(toric)
{}

// =============================================================================
//
// =============================================================================
float
Max::call (float x0, float y0, float x1, float y1)
{
    float dx = fabs(x0-x1);
    float dy = fabs(y0-y1);
    if (is_toric) {
        dx = ((1-dx) < dx) ? (1-dx) : dx;
        dy = ((1-dy) < dy) ? (1-dy) : dy;
    }
    return (dx > dy) ? dx : dy;
}

