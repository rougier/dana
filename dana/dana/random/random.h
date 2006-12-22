//
// Copyright (C) 2006 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.


#ifndef __NEURO_RANDOM_H__
#define __NEURO_RANDOM_H__

#include <boost/random.hpp>

namespace neuro { namespace random {
    
	typedef boost::minstd_rand              base_generator_type;
	typedef boost::uniform_real<>           uniform_distribution_type;
	typedef boost::normal_distribution<>    normal_distribution_type;

	typedef boost::variate_generator 
	    <base_generator_type&, uniform_distribution_type> gen_uniform_type;
	    
	typedef boost::variate_generator
	    <base_generator_type&, normal_distribution_type> gen_normal_type;
		
	//Declare the generator as a global variable for easy reuse
	base_generator_type generator(static_cast<unsigned long>(clock()));

    void seed (unsigned long s);
	float uniform (const float& a, const float& b);
    float normal (const float& mu, const float& sigma);

}} // namespace neuro::random


#endif

