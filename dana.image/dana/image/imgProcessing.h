//
// Copyright (C) 2006 Nicolas Rougier
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

#ifndef __DANA_IMAGE_IMGPROCESSING_H__
#define __DANA_IMAGE_IMGPROCESSING_H__

#include <vector>

template<class SOURCE, class RESULT >
class GaussianPyramid {
public:
	inline GaussianPyramid(SOURCE& src,RESULT& res) {
	    mirage::Vector<2,int> f_dim;
	    // It is better to use a filter with a odd size
	    // If not, there will be a bias in the filtering operation
	    f_dim(11,1);
	    ImageDouble filter;
	    filter.resize(f_dim);
		filter = 0;
	    int j;
	   	for(j = 0 ; j < filter._dimension[0] ; j++)
            {
	    		filter._buffer(j) = exp((-(j-(filter._dimension[0]-1)/2.0)*(j-(filter._dimension[0]-1)/2.0))/(4.0*4.0));
            }

		// If you want to save the 1D-filter used, uncomment the following lines
	    /*
          ImageGray8 img_temp;
          img_temp.resize(filter._dimension);
          Scale<ImageDouble, ImageGray8>(filter,img_temp);
          mirage::img::JPEG::write(img_temp,"filter.jpg",100);
		*/
		
		//*******************************
		//TODO : Clarifier ce qui suit !!!
		//*******************************
		RESULT res_temp,res_temp2;
	    res_temp.resize(src._dimension*0.5);
	    res_temp2 = src;
	    res_temp2.setRescalePolicy(new mirage::BilinearPolicy<RESULT>);
	    res_temp2.rescale(src._dimension*0.5); 
	    OptimConvolution<SOURCE,ImageDouble,RESULT>::convolveX(res_temp2,filter,res_temp);
	    OptimConvolution<SOURCE,ImageDouble,RESULT>::convolveY(res_temp,filter,res_temp2);  
	    res = res_temp2; 
	    // and then rescale the result by a factor 2
        // res.setRescalePolicy(new mirage::BilinearPolicy<RESULT>);
	    //res.rescale(res._dimension*0.5);
	} 
};

template<class SOURCE, class RESULT>
class CenterSurround {

public:
	inline CenterSurround(std::vector<SOURCE> &src, std::vector<RESULT> &res,unsigned int min_level,unsigned int max_level,unsigned int min_delta,unsigned int max_delta,unsigned int size_level)
	{
		// We begin by rescaling all the images to the size of the image at position size_level in the src pyramid
		std::vector<SOURCE> image_temp = src;
		for(unsigned int i = 0 ; i < image_temp.size() ; i++)
            {
                image_temp[i].setRescalePolicy(new mirage::BilinearPolicy<SOURCE>);
                image_temp[i].rescale(image_temp[size_level]._dimension);	
            }
		SOURCE temp;
		temp.resize(image_temp[size_level]._dimension);
		for(unsigned int i = min_level ; i <= max_level & i<image_temp.size() ; i++)
            {
                for(unsigned int j = i + min_delta ; j <= i+max_delta & j < image_temp.size() ; j++)
                    {
                        mirage::algo::BinaryOp<SOURCE,SOURCE,RESULT,absMinus<SOURCE, RESULT> > (image_temp[i],image_temp[j],temp);
                        res.push_back(temp);
                    }
            }
	}	
};

template<class SOURCE1,class SOURCE2, class RESULT>
class computeMask
{
public:
	inline void operator()(typename SOURCE1::value_type& src1,
                           typename SOURCE2::value_type& src2,
                           typename RESULT::value_type& res)
    {
        res = src1 * src2;
    }     
};

#endif
