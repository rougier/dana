#ifndef __GLPYTHON_WORLD_CORE_UTIL_H__
#define __GLPYTHON_WORLD_CORE_UTIL_H__

namespace glpython 
{
    namespace world
        {
            namespace core 
                {
                    
                    class Util
                        {
                            
                        public:
                            static void cross_prod(float u[3], float v[3], float res[3])
                                {
                                    res[0] = u[1]*v[2] - u[2]*v[1];
                                    res[1] = u[2]*v[0] - u[0]*v[2];
                                    res[2] = u[0]*v[1] - u[1]*v[0];
                                };
                            
                            static void normalize(float u[3])
                                {
                                    double norm = sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
                                    u[0] /= norm;
                                    u[1] /= norm;
                                    u[2] /= norm;  
                                };
                        };
                }
        }
}

#endif
