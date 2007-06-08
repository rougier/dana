template<class SOURCE, class RESULT>
class RGBToIntensity
{
public:
	inline void operator()(typename SOURCE::value_type& rgb,
                           typename RESULT::value_type& gray)
    {
        gray = (typename RESULT::value_type) (0.33*rgb._red+0.33*rgb._green+0.33*rgb._blue);
    }    
};


template<class SOURCE, class RESULT>
class RGBToRG
{
public:
	inline void operator()(typename SOURCE::value_type& rgb,
                           typename RESULT::value_type& rg)
    {
		rg = (typename RESULT::value_type) (rgb._red - rgb._blue > 0 ? rgb._red - rgb._blue : 0 );
    }      
};

template<class SOURCE, class RESULT>
class RGBToBY
{
public:
	inline void operator()(typename SOURCE::value_type& rgb,
                           typename RESULT::value_type& by)
    {
		by = (typename RESULT::value_type) ( rgb._blue - ((1-0.5)*rgb._red+ 0.5*rgb._green) > 0 ? rgb._blue -((1-0.5)*rgb._red+ 0.5*rgb._green)  : 0);
    }     
};

template<class SOURCE, class RESULT>
class Normalize
{
public:
	inline void operator()(typename SOURCE::value_type& channel,
						   typename SOURCE::value_type& intensity,
						   typename RESULT::value_type& result)
    {
        result = (typename RESULT::value_type) (intensity == 0 ? 0 : channel / intensity);
    }
};

template<class SOURCE, class RESULT>
class absMinus
{
public:
	inline void operator()(typename SOURCE::value_type& src1,
                           typename SOURCE::value_type& src2,
                           typename RESULT::value_type& res)
    {
        typename SOURCE::value_type value = src1-src2;
        res = (typename RESULT::value_type)(value < 0 ? 0 : value);
    }     
};


template<class SOURCE, class RESULT >
class Normalisation {
public:
	inline Normalisation(SOURCE& src,RESULT& res) {

		double glob_inh = 0.05;
		double ex_amp = 0.5;
		double in_amp = 1.5;
		double ex_sig = 2.0;
		double in_sig = 25.0;		

		// We need the maximum of the map to determine the global inhibition
		typename SOURCE::value_type max;
		typename SOURCE::pixel_type src_pxl,src_pxl_end;
		max = *src.begin();		
		src_pxl_end = src.end();
		for(src_pxl = src.begin();src_pxl != src_pxl_end; ++src_pxl) {
	      	max = *src_pxl > max ? *src_pxl : max;
		}
		
	    mirage::Vector<2,int> f_dim;
	    int w = 20;
	    f_dim(w,1);
	    ImageDouble ex_filter,in_filter;
	    ex_filter.resize(f_dim);
	    in_filter.resize(f_dim);
		ex_filter = 0;
		in_filter = 0;
	    int j;
	   	for(j = 0 ; j < ex_filter._dimension[0] ; j++)
            {
	    		ex_filter._buffer(j) = ex_amp*exp((-(j-(ex_filter._dimension[0]-1)/2.0)*(j-(ex_filter._dimension[0]-1)/2.0))/(ex_sig*ex_sig));
            }
	   	for(j = 0 ; j < in_filter._dimension[0] ; j++)
            {
	    		in_filter._buffer(j) = in_amp*exp((-(j-(in_filter._dimension[0]-1)/2.0)*(j-(in_filter._dimension[0]-1)/2.0))/(in_sig*in_sig));
            }	    

		RESULT res_temp,ex_res,in_res;
		ex_res.resize(res._dimension);
		in_res.resize(res._dimension);
		res_temp.resize(res._dimension);

	    OptimConvolution<SOURCE,ImageDouble,RESULT>::convolveX(src,ex_filter,res_temp);
	    OptimConvolution<SOURCE,ImageDouble,RESULT>::convolveY(res_temp,ex_filter,ex_res);   
	    OptimConvolution<SOURCE,ImageDouble,RESULT>::convolveX(src,in_filter,res_temp);
	    OptimConvolution<SOURCE,ImageDouble,RESULT>::convolveY(res_temp,in_filter,in_res);
	    
	    src_pxl_end = in_res.end();
	  	for(src_pxl = in_res.begin();src_pxl != src_pxl_end; ++src_pxl) {
	      	*src_pxl += glob_inh * max;
		}
	    // and compute the difference
	    mirage::algo::BinaryOp<RESULT,RESULT,RESULT,absMinus<RESULT, RESULT> > (ex_res,in_res,res);
	} 
};

template<class SOURCE, class RESULT >
class Scale {
public:
	inline Scale(SOURCE& src,RESULT& res,double max_value=255.0) {
        // We first determine the minimum and maximum values of the image
        typename SOURCE::pixel_type src_pxl,src_pxl_end;
        typename RESULT::pixel_type res_pxl;
        typename SOURCE::value_type min,max;
        min = *src.begin();
        max = min;
        src_pxl_end = src.end();
        for(src_pxl = src.begin();src_pxl != src_pxl_end; ++src_pxl) {
            min = *src_pxl < min ? *src_pxl : min;
            max = *src_pxl > max ? *src_pxl : max;
        }
        //std::cout << "Max : " << max << " - Min : " << min << std::endl;
        // We then scale the image  
        for(src_pxl = src.begin(),res_pxl = res.begin();src_pxl != src_pxl_end; ++src_pxl,++res_pxl)
            *res_pxl = (typename RESULT::value_type)((max_value/(max-min))*(*src_pxl-min));
	}            
};

template<class SOURCE>
class Util
{
public:
	static typename SOURCE::value_type minimum(SOURCE& src) {
       	typename SOURCE::pixel_type src_pxl,src_pxl_end;
 		typename SOURCE::value_type value = *(src.begin());
		src_pxl_end = src.end();
        for(src_pxl = src.begin(); src_pxl != src_pxl_end; ++src_pxl)
            value = *src_pxl < value ? *src_pxl : value;
        return value;
	}

	static typename SOURCE::value_type maximum(SOURCE& src) {
       	typename SOURCE::pixel_type src_pxl,src_pxl_end;
 		typename SOURCE::value_type value = *(src.begin());
		src_pxl_end = src.end();
        for(src_pxl = src.begin(); src_pxl != src_pxl_end; ++src_pxl)
            value = *src_pxl > value ? *src_pxl : value;
        return value;
	}
};	
