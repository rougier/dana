/***************************************************************************
 *   Copyright (C) 2007 by Jeremy Fix                                      *
 *   Jeremy.Fix@Loria.fr                                                   *
 *   This code is largely inspired from the matlab code of Dirk Walter     *
 *   You can find it at http://www.saliencytoolbox.net                     *
 // //////////////////////////////////////////////////////////////////// //
 // Major portions of the SaliencyToolbox are protected under two        //
 // U.S. patents:                                                        //
 // (1) "Computation of Intrinsic Perceptual Saliency in Visual          //
 //     Environments, and Applications" by Christof Koch and Laurent     //
 //     Itti, California Institute of Technology, 2001 (patent pending;  //
 //     application number 09/912,225; filed July 23, 2001).             //
 // (2) "A system and method for attentional selection" by Ueli          //
 //     Rutishauser, Dirk Walther, Christof Koch, and Pietro Perona,     //
 //     California Institute of Technology, 2004  (patent pending;       //
 //     application number 10/866,311; filed June 10, 2004).             //
 // See http://portal.uspto.gov/external/portal/pair for current status. //
 // //////////////////////////////////////////////////////////////////// //
 * You can access to the first patent at the address :                     *
 *                                                                         *
 *   http://www.freshpatents.com/Computation-of-intrinsic-perceptual-      *
 *   saliency-in-visual-environments-and-                                  *
 *   applications-dt20060928ptan20060215922.php                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <mirage.h>
#include <vector>
#include <math.h>
#include "saliency.h"

using namespace std;
using namespace dana::image;

// Declaration of the static member angle of AngleFiltOp
float AngleFiltOp::angle = 0.0;


Saliency::Saliency()
{
    //constructeur par défaut
    mirage::Init();
	comp_orientation = true;
	comp_color = true;
	comp_sal = false;
	verbose = false;
    comp_save = false;
    pyr_level = 7;
    min_level = 2;
    max_level = 4;
    min_delta = 3;
    max_delta = 4;
    size_level = 4; 
    image_loaded = false;
}

Saliency::Saliency(bool color, bool orientation, bool save=false,bool verb=false)
{
    //constructeur par défaut
    mirage::Init();
    comp_color = color;
    comp_orientation = orientation;
    comp_save = save;
    verbose = verb;
	comp_sal = true;
    pyr_level = 7;
    min_level = 2;
    max_level = 4;
    min_delta = 3;
    max_delta = 4;
    size_level = 4;  
    image_loaded = false;
}

Saliency::~Saliency(void)
{
    // destructeur par défaut
}

void Saliency::read(char * img_filename)
{
    // Read the image with mirage
    mirage::img::JPEG::read(source,img_filename);
    
    // Say that an image is loaded
    image_loaded = true;

    // Init the vectors of results
    intensity_pyr.clear();
    cs_intensity.clear();
    rg_pyr.clear();
    gr_pyr.clear();
    by_pyr.clear();
    yb_pyr.clear();
    cs_rg.clear();
    cs_gr.clear();
    cs_by.clear();
    cs_yb.clear();
    sobels_pyr.clear();
    sobels_cs.clear();
    sobels_sal.clear();
}

void Saliency::process_color(void)
{
    if(verbose) std::cout << "Computing the color opponency channels" << std::endl;
    rg.resize(source._dimension);
    gr.resize(source._dimension);
    by.resize(source._dimension);
    yb.resize(source._dimension);
	
    // It is necesseray to build an image with values of pixel of type double
    // if we don't do so, we loose a lot of information because of the normalisation operator
    if(verbose) std::cout << "Normalizing the opponency channels by the intensity " << std::endl;
    mirage::algo::UnaryOp<ImageRGB24,ImageDouble,RGBToRG<ImageRGB24, ImageDouble> > (source,rg);
    mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,Normalize<ImageDouble, ImageDouble> > (rg,intensity,rg);

    mirage::algo::UnaryOp<ImageRGB24,ImageDouble,RGBToGR<ImageRGB24, ImageDouble> > (source,gr);
    mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,Normalize<ImageDouble, ImageDouble> > (gr,intensity,gr);
		
    mirage::algo::UnaryOp<ImageRGB24,ImageDouble,RGBToBY<ImageRGB24, ImageDouble> > (source,by);		
    mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,Normalize<ImageDouble, ImageDouble> > (by,intensity,by);

    mirage::algo::UnaryOp<ImageRGB24,ImageDouble,RGBToYB<ImageRGB24, ImageDouble> > (source,yb);		
    mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,Normalize<ImageDouble, ImageDouble> > (yb,intensity,yb);
				
    //******************************	
    // Compute the gaussian pyramids
    //******************************
	
    if(verbose) std::cout << "Computing the gaussian pyramid for red/green opponency " << std::endl;
    rg_pyr.push_back(rg);	
    pyr_temp = rg;			
    for(int i = 0 ; i < pyr_level ; i ++)
        {
            GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
            rg_pyr.push_back(pyr_temp);
        }	

    if(verbose) std::cout << "Computing the gaussian pyramid for green/red opponency " << std::endl;
    gr_pyr.push_back(gr);	
    pyr_temp = gr;			
    for(int i = 0 ; i < pyr_level ; i ++)
        {
            GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
            gr_pyr.push_back(pyr_temp);
        }
			
    if(verbose) std::cout << "... for blue/yellow opponency " << std::endl;
    by_pyr.push_back(by);	
    pyr_temp = by;			
    for(int i = 0 ; i < pyr_level ; i ++)
        {
            GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
            by_pyr.push_back(pyr_temp);
        }		
			
    if(verbose) std::cout << "... for yellow/blue opponency " << std::endl;
    yb_pyr.push_back(yb);
    pyr_temp = yb;
    for(int i = 0 ; i < pyr_level ; i ++)
        {
            GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
            yb_pyr.push_back(pyr_temp);
        }
			
    //****************************
    // Compute the center-surround
    //****************************
    // Before the center/surround operation, the images will be rescaled to the size of 
    // the image at position size_level in the pyramid
    if(verbose) std::cout << "Computing the center/surround for color opponency " << std::endl;
    CenterSurround<ImageDouble,ImageDouble> (rg_pyr,cs_rg,min_level,max_level,min_delta,max_delta,size_level); 
    CenterSurround<ImageDouble,ImageDouble> (gr_pyr,cs_gr,min_level,max_level,min_delta,max_delta,size_level);
    CenterSurround<ImageDouble,ImageDouble> (by_pyr,cs_by,min_level,max_level,min_delta,max_delta,size_level); 
    CenterSurround<ImageDouble,ImageDouble> (yb_pyr,cs_yb,min_level,max_level,min_delta,max_delta,size_level); 
            
    if(verbose) std::cout << "Computing the conspicuity maps for color opponency " << std::endl;
    rg_salMap.resize(rg_pyr[size_level]._dimension);
    rg_salMap = 0;		
    for(unsigned int i = 0 ; i < cs_rg.size() ; i ++)
        {
            rg_salMap += cs_rg[i];
        }
    //Normalisation<ImageDouble,ImageDouble>(rg_salMap,rg_salMap);

    gr_salMap.resize(gr_pyr[size_level]._dimension);
    gr_salMap = 0;		
    for(unsigned int i = 0 ; i < cs_gr.size() ; i ++)
        {
            gr_salMap += cs_gr[i];
        }
    //Normalisation<ImageDouble,ImageDouble>(gr_salMap,gr_salMap);
            
    by_salMap.resize(by_pyr[size_level]._dimension);
    by_salMap = 0;			
    for(unsigned int i = 0 ; i < cs_by.size() ; i ++)
        {
            by_salMap += cs_by[i];
        }	
    //Normalisation<ImageDouble,ImageDouble>(by_salMap,by_salMap);		
            
    yb_salMap.resize(yb_pyr[size_level]._dimension);
    yb_salMap = 0;			
    for(unsigned int i = 0 ; i < cs_yb.size() ; i ++)
        {
            yb_salMap += cs_yb[i];
        }	
    //Normalisation<ImageDouble,ImageDouble>(yb_salMap,yb_salMap);	    
}

void Saliency::process_orientation(void)
{
    // Initialisation
    // sobels vector
    if(sobels.size() != orientations.size())
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels.push_back(*(new ImageDouble()));
                    sobels[i] = 0;
                }
        }
    else
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels[i] = 0;
                }
            
        }
    // sobels_pyr
    if(sobels_pyr.size() != orientations.size())
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels_pyr.push_back(*(new std::vector<ImageDouble>()));
                }
            
        }
    else
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels_pyr[i].clear();
                }           
        }
    // sobels_cs
    if(sobels_cs.size() != orientations.size())
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels_cs.push_back(*(new std::vector<ImageDouble>()));
                }
            
        }
    else
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels_cs[i].clear();
                }           
        }
    // sobels_sal
    if(sobels_sal.size() != orientations.size())
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels_sal.push_back(*(new ImageDouble()));
                    sobels_sal[i] = 0;
                }
        }
    else
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobels_sal[i] = 0;
                }
            
        }

    // Calcul

    if(verbose) std::cout << "Computing the orientations' channels" << std::endl;

    sobel_res_h.resize(intensity._dimension);		
    sobel_res_v.resize(intensity._dimension);
    sobel_norm.resize(intensity._dimension);
    mirage::img::Filtering<1,ImageDouble,ImageDouble>::Gray::Sobel::Horizontal(intensity,sobel_res_h); 
    mirage::img::Filtering<1,ImageDouble,ImageDouble,1,1>::Gray::Sobel::Vertical(intensity,sobel_res_v);

    if(verbose) std::cout << "Computing the norm of the Sobel filter" << std::endl;
    mirage::img::Filtering<1,ImageDouble,ImageDouble>::Gray::Sobel::Norm(intensity,sobel_norm);			
	
    if(verbose) std::cout << "Computing the orientation for each pixel" << std::endl;
    sobel_orientations.resize(intensity._dimension);
    mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,AngleOp>(sobel_res_h,sobel_res_v,sobel_orientations);


    for(unsigned int i = 0 ; i < orientations.size() ; i++)
        {
            AngleFiltOp::angle = orientations[i];

            sobels[i].resize(intensity._dimension);
            mirage::algo::UnaryOp<ImageDouble,ImageDouble,AngleFiltOp>(sobel_orientations,sobels[i]);
            mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,computeMask<ImageDouble, ImageDouble, ImageDouble> > (sobels[i],sobel_norm,sobels[i]);            

            sobels_pyr[i].push_back(sobels[i]);
            pyr_temp = sobels[i];
            for(int j = 0 ; j < pyr_level ; j ++)
                {
                    GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
                    sobels_pyr[i].push_back(pyr_temp);
                } 

            CenterSurround<ImageDouble,ImageDouble> (sobels_pyr[i],sobels_cs[i],min_level,max_level,min_delta,max_delta,size_level);

            sobels_sal[i].resize(sobels_pyr[i][size_level]._dimension);
            sobels_sal[i] = 0;
            for(unsigned int j = 0 ; j < sobels_cs[i].size() ; j ++)
                {
                    sobels_sal[i] += sobels_cs[i][j];			
                }	
            //Normalisation<ImageDouble,ImageDouble>(sobels_sal[i],sobels_sal[i]);

        }
    if(sobels_sal.size() >= 1)
        {
            sobel_salMap.resize(sobels_sal[0]._dimension);
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    sobel_salMap += sobels_sal[i];
                }
        }
}

void Saliency::process(void)
{
    if(!(image_loaded))
        {
            std::cerr << "[ERROR] No image has been loaded, you must read one before processing." << std::endl;
            return;
        }
    
    
    //**************************
    // Compute the intensity map
    //**************************
    // The intensity map is the mean of the R, G and B channels of the source image
    if(verbose) std::cout << "Computing the intensity map" << std::endl;
    intensity.resize(source._dimension);
    mirage::algo::UnaryOp<ImageRGB24,ImageDouble,RGBToIntensity<ImageRGB24, ImageDouble> > (source,intensity);
		
    if(verbose) std::cout << "Computing the gaussian pyramid for intensity" << std::endl;	
    intensity_pyr.push_back(intensity);	
    pyr_temp = intensity;			
    for(int i = 0 ; i < pyr_level ; i ++)
        {
            GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
            intensity_pyr.push_back(pyr_temp);
        }
		
    if(verbose) std::cout << "Computing the center/surround for the intensity pyramid" << std::endl;
    CenterSurround<ImageDouble,ImageDouble> (intensity_pyr,cs_intensity,min_level,max_level,min_delta,max_delta,size_level); 

    if(verbose) std::cout << "Computing the conspicuity map for the intensity channel " << std::endl;
    intensity_salMap.resize(intensity_pyr[size_level]._dimension);
    intensity_salMap = 0;
    for(unsigned int i = 0 ; i < cs_intensity.size() ; i ++)
        {
            intensity_salMap += cs_intensity[i];			
        }
    //Normalisation<ImageDouble,ImageDouble>(intensity_salMap,intensity_salMap);

    //*******************************
    // Compute the opponency channels
    //*******************************
		
    if(comp_color)
        {
            process_color();
        }
		
    //*************************
    // Compute the orientations
    //*************************
    if(comp_orientation && (orientations.size() != 0))
        {
            process_orientation();
        }
				
    //********************
    // We save the results
    //********************
    if(comp_save)
        {
            save();
        }
}

void Saliency::save(void)
{
    char filename[50];

    if(verbose) std::cout << "Saving the images " << std::endl;
    mirage::img::JPEG::write(source,"source.jpg",100);
		
    //**** For intensity ****//
    img_temp.resize(intensity_salMap._dimension);
    Scale<ImageDouble, ImageGray8>(intensity_salMap,img_temp);
    mirage::img::JPEG::write(img_temp,"saliency_intensity.jpg",100);			

    //**** For color ****/
    if(comp_color)
        {
            std::cout << "Saving comp_color" <<std::endl;
            
            img_temp.resize(rg_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(rg_salMap,img_temp);
            mirage::img::JPEG::write(img_temp,"saliency_rg.jpg",100);
            
            img_temp.resize(gr_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(gr_salMap,img_temp);
            mirage::img::JPEG::write(img_temp,"saliency_gr.jpg",100);
            
            img_temp.resize(by_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(by_salMap,img_temp);		
            mirage::img::JPEG::write(img_temp,"saliency_by.jpg",100);
            
            img_temp.resize(yb_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(yb_salMap,img_temp);		
            mirage::img::JPEG::write(img_temp,"saliency_yb.jpg",100);            
        }
        
    //**** For the orientations ****//
    if(comp_orientation && orientations.size() != 0)
        {
            // If orientations.size() == 0, the sobel saliency map is not computed
            std::cout << "Saving comp_orientation" <<std::endl;
            for(unsigned int i = 0 ; i < orientations.size() ; i ++)
                {
                    img_temp.resize(sobels_sal[i]._dimension);
                    Scale<ImageDouble, ImageGray8>(sobels_sal[i],img_temp);
                    sprintf(filename, "saliency_sobel_0.%i.jpg",i);
                    mirage::img::JPEG::write(img_temp,filename,100);
                }
        }
}

void Saliency::print_channels()
{
    std::cout << "Available channels : "<<std::endl;
    int index = 0;

    std::cout << "0 - Intensity channel " << std::endl;
    index += 1;
    
    if(comp_color)
        {
            std::cout << "1 - R+G- channel" << std::endl;
            std::cout << "2 - G+R- channel" << std::endl;
            std::cout << "3 - B+Y- channel" << std::endl;
            std::cout << "4 - Y+B- channel" << std::endl;           
            index += 4;
        }
    
    if(comp_orientation)
        {
            for(unsigned int i = 0 ; i < orientations.size() ; i++)
                {
                    // An orientation is supposed to be a multiple of PI
                    std::cout << i+index << " - Orientation : " << orientations[i]/M_PI << " Pi" <<  std::endl;
                }
        }
    
}

void Saliency::set_map(int channel, core::LayerPtr layer)
{
    channel_maps[layer] = channel;    
}

void Saliency::clamp(void)
{
    //TODO : Be sure that the image is processed before enabling clamp

    // We browse the map channel_maps 
    // resize the saliency map
    // and clamp the result in the layer
    ImageDouble tmp_image;
    core::LayerPtr tmp_layer;
    std::map< core::LayerPtr, int >::iterator cur;

    // We first build a vector containing all the channels
    std::vector< ImageDouble* > tmp_channels;
    tmp_channels.push_back(&intensity_salMap);
    if(comp_color)
        {
            tmp_channels.push_back(&rg_salMap);
            tmp_channels.push_back(&gr_salMap);
            tmp_channels.push_back(&by_salMap);
            tmp_channels.push_back(&yb_salMap);
        }
    for(unsigned int i = 0 ; i < sobels_sal.size() ; i++)
        {
            tmp_channels.push_back(&(sobels_sal[i]));
        }
    
    for(cur = channel_maps.begin() ; cur != channel_maps.end() ; cur++)
        {
            tmp_layer = (*cur).first;
            int tmp_width = (tmp_layer->get_map()->width);
            int tmp_height = (tmp_layer->get_map()->height);
            int tmp_index = (*cur).second;
            
            // We get the channel and copy it in tmp
            // If you add channels, be carefull with the index of orientation !!
            

            tmp_image.resize((tmp_channels[tmp_index])->_dimension);
            tmp_image = *(tmp_channels[tmp_index]);

            mirage::Vector<2,int> tmp_dimension;
            tmp_dimension((mirage::Args<2,int>(),tmp_width,tmp_height));

            tmp_image.rescale(tmp_dimension);
 
            Scale<ImageDouble, ImageDouble>(tmp_image,tmp_image,1.0);

            // And clamp the result in the layer
            ImageDouble::pixel_type sal_pxl,sal_pxl_end;
            sal_pxl = tmp_image.begin();
            sal_pxl_end =  tmp_image.end();

            for(int j = 0 ; j < tmp_height ; j++)
                {
                    for(int i = 0 ; i < tmp_width ; i++)
                        {
                            tmp_layer->get(i,j)->potential = *sal_pxl;
                            sal_pxl++;
                        }
                }
        }
}


// ============================================================================
//    Boost wrapping code
// ============================================================================
void
Saliency::boost (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Saliency> >();

    class_<Saliency>("Saliency",
                     "======================================================================\n"
                     "\n"
                     " Saliency map de Itti \n"
                     "======================================================================\n",
                     init<> ("init() : Compute all the channels and save them by default")
                     )
        .def(init<bool,bool,bool,bool>(args("color","orientation","save","verbose"), "init(color,orientation,save,verbose) Specifies which channels to compute, and if we save the results \n"))
        .def("read",&Saliency::read,"Read an image to be processed\n")
        .def("add_orientation",&Saliency::add_orientation,"Add an orientation to compute\n")
        .def("process",&Saliency::process,
             "Processes the input image and produces the conspicuity maps\n")
        .def("print_channels",&Saliency::print_channels,
             "Print the available channels and the index to access to them\n")
        .def("set_map",&Saliency::set_map,
             "Associates a layer to a channel. Use print_channels to display the possible channels\n")
        .def("clamp",&Saliency::clamp,
             "Clamp the results of the image processing in the previously specified maps\n")
        ;
}


