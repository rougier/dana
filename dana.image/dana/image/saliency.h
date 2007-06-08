#ifndef __DANA_IMAGE_SALIENCY_H__
#define __DANA_IMAGE_SALIENCY_H__

#include <boost/python.hpp>
#include "mirage.h"
#include "types.h"
#include "convolution.h"
#include "util.h"
#include "orientations.h"
#include "imgProcessing.h"
#include <vector>

using namespace boost::python;


namespace dana { namespace image { 

class Saliency 
{
 private:
    // Private fields
    ImageRGB24 source;
    ImageDouble intensity;
    ImageDouble rg, by;
    ImageDouble pyr_temp;
    ImageGray8 img_temp;
    ImageGray8 sal;
    // Gaussian and center-surround pyramids and saliency map for the intensity channel
    std::vector<ImageDouble> intensity_pyr;
    std::vector<ImageDouble> cs_intensity;
    ImageDouble intensity_salMap;

    // Gaussian and center-surround pyramids and saliency map for the color opponency channels
    std::vector<ImageDouble> rg_pyr, by_pyr;			
    std::vector<ImageDouble> cs_rg,cs_by;		
    ImageDouble rg_salMap,by_salMap;
		
    // Gaussian pyramid, center-surround and saliency map for the orientation channels
    ImageDouble sobel_res_h,sobel_res_v,sobel_norm,sobel_orientations,sobel_0,sobel_45,sobel_90,sobel_135;
    ImageInt filter_h,filter_v;
    std::vector<ImageDouble> sobel_0_pyr,sobel_45_pyr,sobel_90_pyr,sobel_135_pyr;		
    std::vector<ImageDouble> cs_sobel_0,cs_sobel_45,cs_sobel_90,cs_sobel_135;
    ImageDouble sobel_0_salMap,sobel_45_salMap,sobel_90_salMap,sobel_135_salMap,sobel_salMap;

    // Rescaled,normalised maps used to compute the saliency map
    ImageDouble scaled_intensity,scaled_rg,scaled_by,scaled_sobel,salMap;
    ImageRGB24 saliency;

    int pyr_level;
    unsigned int min_level;
    unsigned int max_level;
    unsigned int min_delta;
    unsigned int max_delta;
    unsigned int size_level; 	
    
    bool comp_orientation,comp_orientation_save,comp_color,comp_color_save,comp_cs,comp_cs_save,comp_sal,verbose;
 
 public:

    Saliency(void);
    Saliency(char * filename);
    ~Saliency(void);
    void process(void);

    static void boost(void);

    
};
 
}}


#endif
