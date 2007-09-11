//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$


#ifndef __GLPYTHON_OBJECTS_VEC3F_H__
#define __GLPYTHON_OBJECTS_VEC3F_H__

namespace glpython { namespace objects {
    class Vec4f {
        public:
            union {float x; float red;};
            union {float y; float green;};
            union {float z; float blue;};
            union {float w; float alpha;};

        public:
            Vec4f (float x=0, float y=0, float z=0, float w=0);
            Vec4f (const Vec4f &other);
            ~Vec4f (void);
        
            void	zero (void);
            void	normalize (void);
            void	scale (const float value);
            float	norm (void);
            float &	operator[] (unsigned int i);
            Vec4f	cross (const Vec4f &other);
            float	dot (const Vec4f &other);
            Vec4f	operator+ (const Vec4f &other);
            Vec4f	operator- (const Vec4f &other);
            Vec4f	operator* (const float value);
            int		operator== (const Vec4f &other);
        };
    }
}

#endif
