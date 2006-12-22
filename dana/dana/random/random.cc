//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.

#include <boost/python.hpp>
#include "random.h"

void
neuro::random::seed(unsigned long s)
{
    generator.seed(s);
}

float
neuro::random::uniform (const float& a, const float& b)
{
    //Set up the desired distribution
    gen_uniform_type ran_gen(generator, uniform_distribution_type(a, b));

    //Get a random number
    return ran_gen();
}

float
neuro::random::normal (const float& mu, const float& sigma)
{
    //Set up the desired distribution
    gen_normal_type ran_gen(generator, normal_distribution_type(mu, sigma));

    //Get a random number
    return ran_gen();
}

BOOST_PYTHON_MODULE(_random)
{
    using namespace boost::python;
    
    def ("seed", neuro::random::seed);
    def ("uniform", neuro::random::uniform);
    def ("normal", neuro::random::normal);
}


