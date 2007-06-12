#ifndef __DANA_IMAGE_SALIENCY_H__
#define __DANA_IMAGE_SALIENCY_H__

#include <boost/python.hpp>
#include "mirage.h"
#include "types.h"
#include "convolution.h"
#include "util.h"
#include "imgProcessing.h"
#include "orientations.h"
#include <vector>
#include <map>

#include "core/layer.h"
#include "core/map.h"
#include "core/unit.h"

using namespace boost::python;


namespace dana { 
    namespace image {

        class Saliency 
            {
            private:
                ///// Tools
                // Is the image buffer loaded ?
                bool image_loaded;

                ///// For the image processing part
                // Private fields
                ImageRGB24 source;
                ImageDouble intensity;
                ImageDouble rg, by,gr,yb;
                ImageDouble pyr_temp;
                ImageGray8 img_temp;
                ImageGray8 sal;
                // Gaussian and center-surround pyramids and saliency map for the intensity channel
                std::vector<ImageDouble> intensity_pyr;
                std::vector<ImageDouble> cs_intensity;
                ImageDouble intensity_salMap;

                // Gaussian and center-surround pyramids and saliency map for the color opponency channels
                std::vector<ImageDouble> rg_pyr, gr_pyr, by_pyr, yb_pyr;
                std::vector<ImageDouble> cs_rg, cs_gr, cs_by, cs_yb;
                ImageDouble rg_salMap, gr_salMap, by_salMap, yb_salMap;
		
                // Gaussian pyramid, center-surround and saliency map for the orientation channels
                std::vector<double> orientations;
    
                ImageDouble sobel_res_h,sobel_res_v,sobel_norm,sobel_orientations;
                std::vector<ImageDouble> sobels;
                ImageInt filter_h,filter_v;
                std::vector<std::vector<ImageDouble> > sobels_pyr;		
                std::vector<std::vector<ImageDouble> > sobels_cs;
                std::vector<ImageDouble> sobels_sal;
                ImageDouble sobel_salMap;
    
                // Rescaled,normalised maps used to compute the saliency map
                ImageDouble scaled_intensity,scaled_rg,scaled_by,scaled_sobel,salMap;
                //ImageRGB24 saliency;

                int pyr_level;
                unsigned int min_level;
                unsigned int max_level;
                unsigned int min_delta;
                unsigned int max_delta;
                unsigned int size_level; 	
    
                bool comp_orientation,comp_save,comp_color,comp_sal,verbose;

                ///// For the dana part
                //std::vector< channel_map > channel_maps;
                std::map< core::LayerPtr, int> channel_maps;
                
 
            public:

                Saliency(void);
                Saliency(bool color,bool orientation, bool save, bool verb);
                ~Saliency(void);
                void read(char * img_filename);
                void add_orientation(double o) 
                    {
                        orientations.push_back(o);
                    };
                void process_color(void);
                void process_orientation(void);
                void process(void);
                void save(void);
                void print_channels(void);

                // dana::Map management
                void set_map(int channel, core::LayerPtr layer);
                void clamp(void);

                static void boost(void);

    
            };
 
    }
}



#endif
