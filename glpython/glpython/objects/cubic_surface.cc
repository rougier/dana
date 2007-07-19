//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include "cubic_surface.h"

using namespace boost::python;
using namespace glpython::objects;


//_________________________________________________________________CubicSurface
CubicSurface::CubicSurface (numeric::array X,
                  tuple frame,
                  core::ColormapPtr colormap,
                  float zscale,
                  float alpha,
                  bool has_grid,
                  bool has_border,
                  std::string name) : Array (X,frame,colormap,alpha,has_grid,has_border,name)

{
    this->zscale = zscale;
	vertices  = 0;
	nvertices = 0;
    colors    = 0;
	indices   = 0;
	nindices  = 0;
}


//________________________________________________________________~CubicSurface
CubicSurface::~CubicSurface (void)
{
 	delete [] vertices;
	delete [] colors;
	delete [] indices;
}

//___________________________________________________________________initialize
void
CubicSurface::initialize (void)
{
    float x, y, w, h;
    try {
        x = extract< float >(this->frame[0])();
        y = extract< float >(this->frame[1])();
        w = extract< float >(this->frame[2])();
        h = extract< float >(this->frame[3])();
    } catch (...) {
        PyErr_Print();
        return;
    }

    Array::initialize();

	nvertices	= 2*d0*2*d1;
	delete [] vertices;
	vertices	= new Vec4f[nvertices];
    delete [] colors;
	colors		= new Vec4f[nvertices];
    nindices	= (2*d0-1)*(2*d1-1)*4;
    delete [] indices;
	indices		= new unsigned int[nindices];

    int index = 0;
	for (int j=0; j < int(2*d1-1); j++) {
        for (int i=0; i < int(2*d0-1); i++) {
            int u = int (j*2*d0 + i);
            indices[index++] = u;
            indices[index++] = u+1;
            indices[index++] = u+2*d0+1;
            indices[index++] = u+2*d0;
        }
    }

    float dx = (1.0f / float(d0))*w;
	float dy = (1.0f / float(d1))*h;  
	for (unsigned int j=0; j < 2*d1; j += 2) {
        for (unsigned int i=0; i < 2*d0; i += 2) {
            vertices[i + j*2*d0].x = x+.5*i*dx-.5;
            vertices[i + j*2*d0].y = y+.5*j*dy-.5;
            
            vertices[i +  j*2*d0 +1].x = x+.5*i*dx + dx-.5;
            vertices[i +  j*2*d0 +1].y = y+.5*j*dy-.5;
            
            vertices[i + (j+1)*2*d0].x = x+.5*i*dx-.5;
            vertices[i + (j+1)*2*d0].y = y+.5*j*dy+dy-.5;
            
            vertices[i + (j+1)*2*d0 +1].x = x+.5*i*dx +dx-.5;
            vertices[i + (j+1)*2*d0 +1].y = y+.5*j*dy +dy-.5;
        }
    }

    update();
}

//_______________________________________________________________________render
void
CubicSurface::render (void)
{
    if (!vertices)
        initialize();
    
    if (dirty)
        update();
    
    GLint mode;
    glGetIntegerv (GL_RENDER_MODE, &mode);

    float x, y, w, h;
    try {
        x = extract< float >(this->frame[0])();
        y = extract< float >(this->frame[1])();
        w = extract< float >(this->frame[2])();
        h = extract< float >(this->frame[3])();
    } catch (...) {
        PyErr_Print();
        return;
    }

    glPushAttrib (GL_ENABLE_BIT);
    glDisable (GL_TEXTURE_RECTANGLE_ARB);
    glPolygonMode (GL_FRONT, GL_FILL);

    if (mode == GL_RENDER) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glColor4f (1,1,1,alpha);
        glPolygonOffset (1.0f, 1.0f);
        glEnable (GL_POLYGON_OFFSET_FILL);
    	glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glEnableClientState (GL_VERTEX_ARRAY);
        glVertexPointer (3, GL_FLOAT, sizeof(Vec4f), &vertices[0].x);
        glEnableClientState (GL_COLOR_ARRAY);
        glColorPointer (4, GL_FLOAT, sizeof(Vec4f), &colors[0].red);
        glDrawElements(GL_QUADS, nindices, GL_UNSIGNED_INT, indices);
        glDisableClientState (GL_COLOR_ARRAY);

        glDisable (GL_POLYGON_OFFSET_FILL);
            
        if (has_grid) {
//            glLineWidth (0.5f);
            glColor4f (0,0,0,.25);
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
            glDrawElements (GL_QUADS, nindices, GL_UNSIGNED_INT, indices);
        }
        glDisableClientState (GL_VERTEX_ARRAY);
        glDisable (GL_BLEND);
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        // Selection
        if ((sx > -1) && (sy > -1)) {
            glColor3f (0,0,0);
            glPointSize (5.0f);
            glBegin (GL_POINTS);
            glVertex3f (vertices[sy*d0+sx].x, vertices[sy*d0+sx].y, vertices[sy*d0+sx].z*1.01);
            glEnd ();
            glPointSize(1.0f);
        }

        // Border
        if (has_border) {
            glColor4f (0,0,0,1);
            glBegin (GL_QUADS);
            glVertex2f (-0.5+x,   -0.5+y);
            glVertex2f (-0.5+x+w, -0.5+y);
            glVertex2f (-0.5+x+w, -0.5+y+h);
            glVertex2f (-0.5+x,   -0.5+y+h);
            glEnd ();
        }

    } else if (mode == GL_SELECT) {
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        for (int j=0; j<int(d1-1); j++) {
            for (int i=0; i<int(d0-1); i++) {
                glLoadName (id + j*d0+i);
                glBegin (GL_QUADS);
                glVertex3f (vertices[j*d0+i].x,       vertices[j*d0+i].y,       vertices[j*d0+i].z);
                glVertex3f (vertices[j*d0+i+1].x,     vertices[j*d0+i+1].y,     vertices[j*d0+i+1].z);
                glVertex3f (vertices[(j+1)*d0+i+1].x, vertices[(j+1)*d0+i+1].y, vertices[(j+1)*d0+i+1].z);
                glVertex3f (vertices[(j+1)*d0+i].x,   vertices[(j+1)*d0+i].y,   vertices[(j+1)*d0+i].z);
                glEnd ();
            }
        }
        glLoadName (0);  
    }
    glPopAttrib ();

}

//_______________________________________________________________________update
void
CubicSurface::update (void)
{
    float d[4];
    Vec4f color, wcolor;
    int jj = 0;
    int s = 4*sizeof(float);
    float v;
    for (unsigned int j=0; j<2*d1; j += 2) {
        int ii = 0;
        for (unsigned int i=0; i<2*d0; i += 2) {
            v = *(float *)(this->array->data + jj + ii);
            memcpy (d, this->colormap->color(v).data, s);

            int index = i + j*2*d0;
            vertices[index].z = v*zscale;
            colors[index] = Vec4f(d[0],d[1],d[2]);
            colors[index].alpha  = d[3]*alpha;

            index = i + j*2*d0+1;
            vertices[index].z = v*zscale;
            colors[index] = Vec4f(d[0],d[1],d[2]);
            colors[index].alpha  = d[3]*alpha;

            index = i + (j+1)*2*d0;
            vertices[index].z = v*zscale;
            colors[index] = Vec4f(d[0],d[1],d[2]);
            colors[index].alpha  = d[3]*alpha;

            index = i + (j+1)*2*d0 + 1;
            vertices[index].z = v*zscale;
            colors[index] = Vec4f(d[0],d[1],d[2]);
            colors[index].alpha  = d[3]*alpha;

            ii += array->strides[1];
        }
        jj += array->strides[0];
    }
}

//________________________________________________________________python_export
void
CubicSurface::python_export (void)
{

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<CubicSurface> >();
    import_array();
    numeric::array::set_module_and_type ("numpy", "ndarray");  

    class_<CubicSurface, bases< Array> > ("CubicSurface",
    " ______________________________________________________________________\n"
    "                                                                       \n"
    " ______________________________________________________________________\n",
    init<numeric::array,
         optional <tuple, core::ColormapPtr, float, float, bool, bool, std::string> > (
        (arg("X"),
         arg("frame") = make_tuple (0,0,1,1),
         arg("cmap")  = core::Colormaps::Default,
         arg("zscale") = 0.25f,
         arg("alpha") = 1,
         arg("has_grid") = true,
         arg("has_border") = true,
         arg("name") = "CubicSurface"),
        "__init__ ( X, frame, cmap, zscale, alpha, has_grid, name )\n"))
    ;
}
