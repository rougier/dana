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

#ifndef __DANA_IMAGE_ORIENTATIONS_H__
#define __DANA_IMAGE_ORIENTATIONS_H__

namespace dana 
{
    namespace image
        {
            class NormOp 
                {
                public:
                    inline void operator()(double& hor,double& vert,double& norm) {
                        norm  = (sqrt((hor*hor + vert*vert)/16.0));
                    };
                };    

            class AngleOp 
                {
                public:
                    inline void operator()(double& hor,double& vert,double& angle) {
                        //atan retourne une valeur entre -pi/2 et pi /2
                        //TODO : vérifier si ce qui suit est correct
                        if(hor == 0.0)
                            angle = -M_PI;
                        else
                            angle  = atan(vert/hor);
                    };
                }; 

            class AngleFiltOp 
                {
                public:
                    static float angle;

                    inline void operator()(double& angle_src, double& angle_dst)
                        {
                            if(angle_src == -M_PI)
                                angle_dst = 0;
                            else
                                {
                                    double value = (1.0+cos(2.0*(angle_src-angle)))/2.0;
                                    angle_dst = exp( - (value-1.0)*(value-1.0)/(0.1*0.1));
                                }
                        };
                };
        }
}

#endif
