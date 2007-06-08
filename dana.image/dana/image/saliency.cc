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


Saliency::Saliency()
{
    // Constructeur par défaut:
    std::cerr<<"Never use the default constructor of saliency !"<<std::endl;
    min_level = 2;
    max_level = 4;
    min_delta = 3;
    max_delta = 4;
    size_level = 4; 
}


Saliency::Saliency(char * img_filename)
{
    //constructeur par défaut
    mirage::Init();
    mirage::img::JPEG::read(source,img_filename);
	comp_orientation = true;
    comp_orientation_save = true;
	comp_color = true;
    comp_color_save = true;
	comp_cs = true;
    comp_cs_save = true;
	comp_sal = true;
	verbose = true;
    pyr_level = 7;
    min_level = 2;
    max_level = 4;
    min_delta = 3;
    max_delta = 4;
    size_level = 4; 	
    
}

Saliency::~Saliency(void)
{
    // destructeur par défaut
}

void Saliency::process(void)
{
    char filename[50];
    
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
    Normalisation<ImageDouble,ImageDouble>(intensity_salMap,intensity_salMap);

    //*******************************
    // Compute the opponency channels
    //*******************************
		
    if(comp_color)
        {
            if(verbose) std::cout << "Computing the color opponency channels" << std::endl;
            rg.resize(source._dimension);
            by.resize(source._dimension);
	
            // It is necesseray to build an image with values of pixel of type double
            // if we don't do so, we loose a lot of information because of the normalisation operator
            if(verbose) std::cout << "Normalizing the opponency channels by the intensity " << std::endl;
            mirage::algo::UnaryOp<ImageRGB24,ImageDouble,RGBToRG<ImageRGB24, ImageDouble> > (source,rg);
            mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,Normalize<ImageDouble, ImageDouble> > (rg,intensity,rg);
	
            mirage::algo::UnaryOp<ImageRGB24,ImageDouble,RGBToBY<ImageRGB24, ImageDouble> > (source,by);		
            mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,Normalize<ImageDouble, ImageDouble> > (by,intensity,by);
			
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
			
            if(verbose) std::cout << "... for blue/yellow opponency " << std::endl;
            by_pyr.push_back(by);	
            pyr_temp = by;			
            for(int i = 0 ; i < pyr_level ; i ++)
                {
                    GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
                    by_pyr.push_back(pyr_temp);
                }		
			
            //****************************
            // Compute the center-surround
            //****************************
            if(comp_cs)
                {
                    // Before the center/surround operation, the images will be rescaled to the size of 
                    // the image at position size_level in the pyramid
                    if(verbose) std::cout << "Computing the center/surround for color opponency " << std::endl;
                    CenterSurround<ImageDouble,ImageDouble> (rg_pyr,cs_rg,min_level,max_level,min_delta,max_delta,size_level); 
                    CenterSurround<ImageDouble,ImageDouble> (by_pyr,cs_by,min_level,max_level,min_delta,max_delta,size_level); 
				
                    if(verbose) std::cout << "Computing the conspicuity maps for color opponency " << std::endl;
                    rg_salMap.resize(rg_pyr[size_level]._dimension);
                    rg_salMap = 0;		
                    for(unsigned int i = 0 ; i < cs_rg.size() ; i ++)
                        {
                            rg_salMap += cs_rg[i];			
                        }
                    Normalisation<ImageDouble,ImageDouble>(rg_salMap,rg_salMap);
				
                    by_salMap.resize(by_pyr[size_level]._dimension);
                    by_salMap = 0;			
                    for(unsigned int i = 0 ; i < cs_by.size() ; i ++)
                        {
                            by_salMap += cs_by[i];			
                        }	
                    Normalisation<ImageDouble,ImageDouble>(by_salMap,by_salMap);		
                }
        }
		
    //*************************
    // Compute the orientations
    //*************************
    if(comp_orientation)
        {
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
	
            if(verbose) std::cout << "Computing the 0° channel" << std::endl;
            sobel_0.resize(intensity._dimension);
            mirage::algo::UnaryOp<ImageDouble,ImageDouble,Angle0Op>(sobel_orientations,sobel_0);
            mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,computeMask<ImageDouble, ImageDouble, ImageDouble> > (sobel_0,sobel_norm,sobel_0);
			
            if(verbose) std::cout << "Computing the 45° channel" << std::endl;
            sobel_45.resize(intensity._dimension);
            mirage::algo::UnaryOp<ImageDouble,ImageDouble,Angle45Op>(sobel_orientations,sobel_45);
            mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,computeMask<ImageDouble, ImageDouble, ImageDouble> > (sobel_45,sobel_norm,sobel_45);
			
            if(verbose) std::cout << "Computing the 90° channel" << std::endl;
            sobel_90.resize(intensity._dimension);
            mirage::algo::UnaryOp<ImageDouble,ImageDouble,Angle90Op>(sobel_orientations,sobel_90);
            mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,computeMask<ImageDouble, ImageDouble, ImageDouble> > (sobel_90,sobel_norm,sobel_90);
	
            if(verbose) std::cout << "Computing the 135° channel" << std::endl;
            sobel_135.resize(intensity._dimension);
            mirage::algo::UnaryOp<ImageDouble,ImageDouble,Angle135Op>(sobel_orientations,sobel_135);
            mirage::algo::BinaryOp<ImageDouble,ImageDouble,ImageDouble,computeMask<ImageDouble, ImageDouble, ImageDouble> > (sobel_135,sobel_norm,sobel_135);
			
            // We now compute the gaussian pyramids for each orientation
            if(verbose) std::cout << "Computing the gaussian pyramids..." << std::endl;
            if(verbose) std::cout << "... for orientation 0° " << std::endl;		
            sobel_0_pyr.push_back(sobel_0);	
            pyr_temp = sobel_0;			
            for(int i = 0 ; i < pyr_level ; i ++)
                {
                    GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
                    sobel_0_pyr.push_back(pyr_temp);
                }
            if(verbose) std::cout << "... for orientation 45° " << std::endl;		
            sobel_45_pyr.push_back(sobel_45);	
            pyr_temp = sobel_45;			
            for(int i = 0 ; i < pyr_level ; i ++)
                {
                    GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
                    sobel_45_pyr.push_back(pyr_temp);
                }
				
            if(verbose) std::cout << "... for orientation 90° " << std::endl;		
            sobel_90_pyr.push_back(sobel_90);	
            pyr_temp = sobel_90;			
            for(int i = 0 ; i < pyr_level ; i ++)
                {
                    GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
                    sobel_90_pyr.push_back(pyr_temp);
                }	
			
            if(verbose) std::cout << "... for orientation 135° " << std::endl;		
            sobel_135_pyr.push_back(sobel_135);	
            pyr_temp = sobel_135;			
            for(int i = 0 ; i < pyr_level ; i ++)
                {
                    GaussianPyramid<ImageDouble, ImageDouble> (pyr_temp, pyr_temp);
                    sobel_135_pyr.push_back(pyr_temp);
                }
						
            // And the center/surround for each pyramid
            if(comp_cs)
                {
                    if(verbose) std::cout << "Computing the center/surround for orientations' channels" << std::endl;
                    CenterSurround<ImageDouble,ImageDouble> (sobel_0_pyr,cs_sobel_0,min_level,max_level,min_delta,max_delta,size_level); 
                    CenterSurround<ImageDouble,ImageDouble> (sobel_45_pyr,cs_sobel_45,min_level,max_level,min_delta,max_delta,size_level); 
                    CenterSurround<ImageDouble,ImageDouble> (sobel_90_pyr,cs_sobel_90,min_level,max_level,min_delta,max_delta,size_level); 
                    CenterSurround<ImageDouble,ImageDouble> (sobel_135_pyr,cs_sobel_135,min_level,max_level,min_delta,max_delta,size_level);
				
                    if(verbose) std::cout << "Computing the conspicuity maps for orientations' channels" << std::endl;
                    sobel_0_salMap.resize(sobel_0_pyr[size_level]._dimension);
                    sobel_0_salMap = 0;			
                    for(unsigned int i = 0 ; i < cs_sobel_0.size() ; i ++)
                        {
                            sobel_0_salMap += cs_sobel_0[i];			
                        }	
                    Normalisation<ImageDouble,ImageDouble>(sobel_0_salMap,sobel_0_salMap);		
				
                    sobel_45_salMap.resize(sobel_45_pyr[size_level]._dimension);
                    sobel_45_salMap = 0;			
                    for(unsigned int i = 0 ; i < cs_sobel_45.size() ; i ++)
                        {
                            sobel_45_salMap += cs_sobel_45[i];			
                        }	
                    Normalisation<ImageDouble,ImageDouble>(sobel_45_salMap,sobel_45_salMap);	
		
                    sobel_90_salMap.resize(sobel_90_pyr[size_level]._dimension);
                    sobel_90_salMap = 0;			
                    for(unsigned int i = 0 ; i < cs_sobel_90.size() ; i ++)
                        {
                            sobel_90_salMap += cs_sobel_90[i];			
                        }	
                    Normalisation<ImageDouble,ImageDouble>(sobel_90_salMap,sobel_90_salMap);	
		
                    sobel_135_salMap.resize(sobel_90_pyr[size_level]._dimension);
                    sobel_135_salMap = 0;			
                    for(unsigned int i = 0 ; i < cs_sobel_135.size() ; i ++)
                        {
                            sobel_135_salMap += cs_sobel_135[i];			
                        }	
                    Normalisation<ImageDouble,ImageDouble>(sobel_135_salMap,sobel_135_salMap);	
                        
                    sobel_salMap.resize(sobel_0_salMap._dimension);
                    sobel_salMap = 	sobel_0_salMap + sobel_45_salMap + sobel_90_salMap+ sobel_135_salMap;		
                } 
        }	

    if(comp_sal)
        {
            // Pour calculer la carte de saillance, on commence par redimensionner toutes les images
            // et par les normaliser entre 0 et 255
            scaled_intensity.resize(intensity_salMap._dimension);
            scaled_rg.resize(intensity_salMap._dimension);
            scaled_by.resize(intensity_salMap._dimension);
            scaled_sobel.resize(intensity_salMap._dimension);
            Scale<ImageDouble, ImageDouble>(intensity_salMap,scaled_intensity);
            Scale<ImageDouble, ImageDouble>(rg_salMap,scaled_rg);
            Scale<ImageDouble, ImageDouble>(by_salMap,scaled_by);
            Scale<ImageDouble, ImageDouble>(sobel_salMap,scaled_sobel);
		
            // On les additionne pour former la carte de saillance globale
            salMap.resize(intensity_salMap._dimension);
            salMap = scaled_intensity + scaled_rg + scaled_by + scaled_sobel;
        }

						
    //********************
    // We save the results
    //********************
    // The images are scaled from (pix_value_min,pix_value_max) to [0..255] and converted to ImageGray8 to be saved		 	
    if(verbose) std::cout << "Saving the images " << std::endl;
    mirage::img::JPEG::write(source,"source.jpg",100);
		
    //**** For intensity ****//
    img_temp.resize(intensity._dimension);
    Scale<ImageDouble, ImageGray8>(intensity,img_temp);
    mirage::img::JPEG::write(img_temp,"intensity.jpg",100);

    for(int i = 0 ; i <= pyr_level ; i ++)
        {
            img_temp.resize(intensity_pyr[i]._dimension);
            Scale<ImageDouble, ImageGray8>(intensity_pyr[i],img_temp);
            sprintf(filename, "pyr_intensity.%i.jpg",i);
            mirage::img::JPEG::write(img_temp,filename,100);			
        }
		
    for(unsigned int i = 0 ; i < cs_intensity.size() ; i ++)
        {
            img_temp.resize(cs_intensity[i]._dimension);
            Scale<ImageDouble, ImageGray8>(cs_intensity[i],img_temp);
            sprintf(filename, "cs_intensity.%i.jpg",i);
            mirage::img::JPEG::write(img_temp,filename,100);			
        }	

    img_temp.resize(intensity_salMap._dimension);
    Scale<ImageDouble, ImageGray8>(intensity_salMap,img_temp);
    mirage::img::JPEG::write(img_temp,"saliency_intensity.jpg",100);			

    //**** For color ****/
    if(comp_color_save)
        {
            img_temp.resize(rg._dimension);
            Scale<ImageDouble, ImageGray8>(rg,img_temp);
            mirage::img::JPEG::write(img_temp,"red_green.jpg",100);
			
            img_temp.resize(by._dimension);
            Scale<ImageDouble, ImageGray8>(by,img_temp);
            mirage::img::JPEG::write(img_temp,"blue_yellow.jpg",100);
                    
            for(int i = 0 ; i <= pyr_level ; i ++)
                {
                    img_temp.resize(rg_pyr[i]._dimension);
                    Scale<ImageDouble, ImageGray8>(rg_pyr[i],img_temp);
                    sprintf(filename, "pyr_rg.%i.jpg",i);
                    mirage::img::JPEG::write(img_temp,filename,100);			
                }
            for(int i = 0 ; i <= pyr_level ; i ++)
                {
                    img_temp.resize(by_pyr[i]._dimension);
                    Scale<ImageDouble, ImageGray8>(by_pyr[i],img_temp);
                    sprintf(filename, "pyr_by.%i.jpg",i);
                    mirage::img::JPEG::write(img_temp,filename,100);			
                }
			
            if(comp_cs_save)
                {
                    for(unsigned int i = 0 ; i < cs_rg.size() ; i ++)
                        {
                            img_temp.resize(cs_rg[i]._dimension);
                            Scale<ImageDouble, ImageGray8>(cs_rg[i],img_temp);
                            sprintf(filename, "cs_rg.%i.jpg",i);
                            mirage::img::JPEG::write(img_temp,filename,100);			
                        }	
                    for(unsigned int i = 0 ; i < cs_by.size() ; i ++)
                        {
                            img_temp.resize(cs_by[i]._dimension);
                            Scale<ImageDouble, ImageGray8>(cs_by[i],img_temp);
                            sprintf(filename, "cs_by.%i.jpg",i);
                            mirage::img::JPEG::write(img_temp,filename,100);			
                        }			
                }
        }
        
    if(comp_sal)
        {
            img_temp.resize(rg_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(rg_salMap,img_temp);
            mirage::img::JPEG::write(img_temp,"saliency_rg.jpg",100);
                
            img_temp.resize(by_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(by_salMap,img_temp);
            mirage::img::JPEG::write(img_temp,"saliency_by.jpg",100);
        }
            

    //**** For the orientations ****//
    if(comp_orientation_save)
        {
            img_temp.resize(sobel_0_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(sobel_0_salMap,img_temp);		
            mirage::img::JPEG::write(img_temp,"sobel_0.jpg",100);	
				
            img_temp.resize(sobel_45_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(sobel_45_salMap,img_temp);		
            mirage::img::JPEG::write(img_temp,"sobel_45.jpg",100);			
				
            img_temp.resize(sobel_90_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(sobel_90_salMap,img_temp);		
            mirage::img::JPEG::write(img_temp,"sobel_90.jpg",100);	
				
            img_temp.resize(sobel_135_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(sobel_135_salMap,img_temp);		
            mirage::img::JPEG::write(img_temp,"sobel_135.jpg",100);

            for(int i = 0 ; i <= pyr_level ; i ++)
                {
                    img_temp.resize(sobel_0_pyr[i]._dimension);
                    Scale<ImageDouble, ImageGray8>(sobel_0_pyr[i],img_temp);
                    sprintf(filename, "pyr_sobel_0.%i.jpg",i);
                    mirage::img::JPEG::write(img_temp,filename,100);	

                    img_temp.resize(sobel_45_pyr[i]._dimension);
                    Scale<ImageDouble, ImageGray8>(sobel_45_pyr[i],img_temp);
                    sprintf(filename, "pyr_sobel_45.%i.jpg",i);
                    mirage::img::JPEG::write(img_temp,filename,100);

                    img_temp.resize(sobel_90_pyr[i]._dimension);
                    Scale<ImageDouble, ImageGray8>(sobel_90_pyr[i],img_temp);
                    sprintf(filename, "pyr_sobel_90.%i.jpg",i);
                    mirage::img::JPEG::write(img_temp,filename,100);

                    img_temp.resize(sobel_135_pyr[i]._dimension);
                    Scale<ImageDouble, ImageGray8>(sobel_135_pyr[i],img_temp);
                    sprintf(filename, "pyr_sobel_135.%i.jpg",i);
                    mirage::img::JPEG::write(img_temp,filename,100);		
                }
            if(comp_cs_save)
                {
                    for(unsigned int i = 0 ; i < cs_sobel_0.size() ; i ++)
                        {
                            img_temp.resize(cs_sobel_0[i]._dimension);
                            Scale<ImageDouble, ImageGray8>(cs_sobel_0[i],img_temp);
                            sprintf(filename, "cs_sobel_0.%i.jpg",i);
                            mirage::img::JPEG::write(img_temp,filename,100);

                            img_temp.resize(cs_sobel_45[i]._dimension);
                            Scale<ImageDouble, ImageGray8>(cs_sobel_45[i],img_temp);
                            sprintf(filename, "cs_sobel_45.%i.jpg",i);
                            mirage::img::JPEG::write(img_temp,filename,100);

                            img_temp.resize(cs_sobel_90[i]._dimension);
                            Scale<ImageDouble, ImageGray8>(cs_sobel_90[i],img_temp);
                            sprintf(filename, "cs_sobel_90.%i.jpg",i);
                            mirage::img::JPEG::write(img_temp,filename,100);

                            img_temp.resize(cs_sobel_135[i]._dimension);
                            Scale<ImageDouble, ImageGray8>(cs_sobel_135[i],img_temp);
                            sprintf(filename, "cs_sobel_135.%i.jpg",i);
                            mirage::img::JPEG::write(img_temp,filename,100);			
                        }	
                }

        }
    if(comp_sal)
        {
            img_temp.resize(sobel_salMap._dimension);
            Scale<ImageDouble, ImageGray8>(sobel_salMap,img_temp);
            mirage::img::JPEG::write(img_temp,"saliency_sobel.jpg",100);
        }

    //**** For the saliency map ****//
    if(comp_sal)
        {
            // On commence par sauvegarder la carte de saillance 
            img_temp.resize(salMap._dimension);
            Scale<ImageDouble, ImageGray8>(salMap,img_temp);
            mirage::img::JPEG::write(img_temp,"saliency_gray.jpg",100);
                
            // On la redimensionne à la taille de l'image d'origine
            salMap.setRescalePolicy(new mirage::BilinearPolicy<ImageDouble>);
            salMap.rescale(source._dimension);

            // On la normalise entre 0 et 1
            Scale<ImageDouble, ImageDouble>(salMap,salMap,1.0);
                
            // Et on l'applique comme filtre sur l'image source
            saliency.resize(source._dimension);
            mirage::algo::BinaryOp<ImageRGB24,ImageDouble,ImageRGB24,computeMask<ImageRGB24, ImageDouble, ImageRGB24> > (source,salMap,saliency);
            mirage::img::JPEG::write(saliency,"saliency.jpg",100);
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
                     init<>(
                            "__init__ () -- initialize unit\n")
                     )
        .def(init<char *>())
        .def("process",&Saliency::process,
             "Processes the input image and produces the conspicuity maps\n")
        ;
}


