#include <math.h>

typedef mirage::algo::OptimizedConvolution<ImageDouble,ImageInt,ImageDouble,  
                                           mirage::algo::SumProduct<ImageDouble::value_type,
                                                                    ImageDouble::value_type,
                                                                    ImageInt::value_type > >
OptimizedConvolution;

int inline min(int a, int b)
{
   return (a < b) ? a : b;
}

int inline max(int a, int b)
{
   return (a > b) ? a : b;
}


template<class SOURCE,class FILTER,class RESULT>
class OptimConvolution {
public:
	// This class is an optimized version of the convolution product
	// in the case when the kernel is separable
	static inline void convolveX(SOURCE& src, FILTER& filt, RESULT& res)
	{
		// We convolve the image along the columns
		// ex:
		//    filter : [ f0 , f1 , f2 , f3 , f4 , f5 ]
		//                    ^     ^         ^
		//                   min_f  fp        max_f      -- fp is defined as (wf-1)/2
		//    image (first row):  [ im00, im01 , im02 , im03 , im04 , im05 ] 
 		//                          ^            ^      ^
 		//                          min_j        jp     max_j
 		// The convolution for the result image at position (0,2) will be computed as :
 		//   im04 * f0 + im03*f3 + im02 * f2 + im01	* f1
 		// There are several problems to cope with :
 		//        - the border effects (out of bounds for the image and normalisation):
 		//             We then define imin and imax to get ride of the out of bound effect
 		//             and sum the filter values to normalise the result
 		//        - the filter can be bigger than the image, so, for the for loops, we have to define low and up boundaries
 		
		if(src._dimension == res._dimension)
            {
                int ws = src._dimension[0];
                int hs = src._dimension[1];
                int wf = filt._dimension[0];

                int i,j,k;
                double val,sum;
                int fp = (int)((wf - 1.0) / 2.0);
                int min_f,max_f;
			 
                for(i = 0 ; i < hs ; i++)
                    {
                        for(j = 0 ; j < ws; j++)
                            {
                                // For all the convolutions, we will always have the term : fp * jp
                                //min_f = max(indice minimal dans le filtre,position centrale moins le nombre de points qu'il reste à droite de j dans la source)
                                min_f = max(0,fp-(ws-j-1));
                                //max_f = min(indice maximal dans le filtre, position centrale plus le nombre de points à gauche de j dans la source);
                                max_f = min(wf-1,fp+j);

                                val = 0.0;
                                sum = 0.0;
                                for(k = min_f ; k < max_f ; k++)
                                    {
                                        val += src._buffer(i*ws + j + fp - k) * filt._buffer(k);
                                        //std::cout << "value : " << val << std::endl;
                                        sum += filt._buffer(k);
                                    }
                                if(sum != 0.0)
                                    res._buffer(i*ws+j) = val / sum;
                                else
                                    res._buffer(i*ws+j) = 0.0;
                            }
                    }
            }
		else
            {
                // src and res images don't have the same size !
                std::cerr << "Error : OptimConvolution::convolveX: The source and result images must be of the same size !" << std::endl;	
            }
	}	
	
	static inline void convolveY(SOURCE& src, FILTER& filt, RESULT &res)
	{
		if(src._dimension == res._dimension)
            {
                int ws = src._dimension[0];
                int hs = src._dimension[1];
                int wf = filt._dimension[0];

                int i,j,k;
                double val,sum; 
                int fp = (int)((wf - 1.0) / 2.0);
                int min_f,max_f;		
                for(j = 0 ; j < ws ; j++)
                    {
                        for(i = 0 ; i < hs; i++)
                            {
                                // For all the convolutions, we will always have the term : fp * ip
                                //min_f = max(indice minimal dans le filtre,position centrale moins le nombre de points qu'il reste à droite de j dans la source)
                                min_f = max(0,fp-(hs-i-1));
                                //max_f = min(nombre de points restants à gauche de j, nombre de points restants à droite dans le filtre);
                                max_f = min(wf-1,fp+i);
                                val = 0.0;
                                sum = 0.0;
                                for(k = min_f ; k < max_f ; k++)
                                    {
                                        val += src._buffer((i+fp-k)*ws + j) * filt._buffer(k);
                                        sum += filt._buffer(k);
                                    }
                                if(sum != 0.0)
                                    res._buffer(i*ws+j) = val / sum;
                                else
                                    res._buffer(i*ws+j) = 0.0;
                            }
                    }			
            }
		else
            {
                // src and res images don't have the same size !
                std::cerr << "Error : OptimConvolution::convolveX: The source and result images must be of the same size !" << std::endl;	
            }
	}
	
};
