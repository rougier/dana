//
// Copyright (C) 2007 Nicolas Rougier - Jeremy Fix
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#define GL_GLEXT_PROTOTYPES

#include <GL/glu.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <iostream>
#include "viewport.h"

#include <fstream>

using namespace glpython::world::core;


//_____________________________________________________________________Viewport
Viewport::Viewport (tuple size, tuple position, bool has_border,
                    bool is_ortho, std::string name) : glpython::core::Viewport (size,position,has_border,is_ortho,name)
{
    observer = glpython::core::ObserverPtr (new Observer("Freefly camera"));
    Observer * obs = dynamic_cast< world::core::Observer *>(observer.get());
    obs->position[0] = 10.0f;
    obs->position[1] = 0.0f;
    obs->position[2] = 0.0f;
    obs->theta = 180;
    obs->phi = 0;
    obs->VectorsFromAngles();

    obs->allow_movement = true;

    button_pressed = false;
}

//____________________________________________________________________~Viewport
Viewport::~Viewport (void)
{
}

//_________________________________________________________________________save

void Viewport::save(char * filename)
{
    GLint viewport [4];
    glGetIntegerv (GL_VIEWPORT, viewport);

    int x0 = geometry[0];
    int y0 = geometry[1];
  
    int larg = geometry[2]; if (larg & 1) ++larg; //Pour avoir des valeurs paires
    int haut = geometry[3]; if (haut & 1) ++haut;

    std::ofstream file (filename, std::ios::out|std::ios::binary);
    if (! file)
        std::cerr << "Cannot open file \"" << filename << "\"!\n";
    else
        {
            file << "P6\n#\n" << larg << ' ' << haut << "\n255\n";
  
            glPixelStorei (GL_UNPACK_ALIGNMENT, 1);
            char * pixels = new char [3 * larg * haut];
            glReadPixels (x0, y0, larg, haut, GL_RGB, GL_UNSIGNED_BYTE, pixels);
  
            char * ppix = pixels + (3 * larg * (haut - 1));
            for (unsigned int j = haut; j-- ; ppix -= 3 * larg) file.write (ppix, 3 * larg);
  
            delete[] pixels;
            file.close ();
        }
}



// void
// Viewport::save (char * filename)
// {
//     // TODO : zoom est un argument de Figure::save(filename,zoom)
//     int zoom = 1;

//     GLint viewport[4];

//     glGetIntegerv( GL_VIEWPORT, viewport );

//     // x,y,w,h = geometry[0..3]
//     int width = 512;//int(geometry[2]*zoom);
//     int height = 256;//int(geometry[3]*zoom);
//     unsigned long bits[height][width];
//     // Setup framebuffer

//     GLuint framebuffer;
//     glGenFramebuffersEXT(1, &framebuffer); 
//     glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);
    
//     // Setup depthbuffer
    
//     GLuint depthbuffer;
//     glGenRenderbuffersEXT(1, &depthbuffer);
//     glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthbuffer);
//     glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
    
//     // Create texture to render to
//     GLuint texture;
//     glGenTexturesEXT(1, &texture);
//     glBindTextureEXT(GL_TEXTURE_2D, texture);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, bits);

//     glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture, 0);
//     glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthbuffer);
    
//     GLuint status;
//     status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
//     if(status != GL_FRAMEBUFFER_COMPLETE_EXT)
//         {
//             std::cerr << "Error in framebuffer activation " << std::endl;
//             return;
//         }
    
//     // Render and save
//     glViewport(0, 0, width, height);
//     glClearColor(1, 1, 1, 1);
//     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//     resize_event(0, 0, width, height);
//     glViewport(0, 0, width, height);
//     render();
    
//     glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, bits);
//     printf("Height, width : %i, %i \n" , width, height);
    

//     Magick::Blob image_data(bits,3*width*height);
//     Magick::Image image(Magick::Geometry(width,height), Magick::ColorRGB(1,1,1));
//     //image.magick("RGBA");
//     //image.magick( "RGBA" );
//     image.read( image_data);
//     image.write(filename);

//     // Cleanup
//     glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
//     glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
//     glDeleteTexturesEXT(GL_TEXTURE_2D, &texture);
//     glDeleteFramebuffersEXT(1,&framebuffer);
//     glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
//     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//     resize_event(viewport[0], viewport[1], viewport[2], viewport[3]);
    
//     printf("File saved to %s \n" , filename);
// }

//______________________________________________________________key_press_event
void
Viewport::key_press_event (std::string key)
{
    if ((!visible) || (!has_focus))
        return;
    Observer * obs = dynamic_cast< world::core::Observer *>(observer.get());
    if(obs->allow_movement)
        {
            
            if(key == "up")
                {
                    obs->position[0] += 0.1*obs->forward[0];
                    obs->position[1] += 0.1*obs->forward[1];
                    obs->position[2] += 0.1*obs->forward[2];

                    obs->look_at[0] += 0.1*obs->forward[0];
                    obs->look_at[1] += 0.1*obs->forward[1];
                    obs->look_at[2] += 0.1*obs->forward[2];
                }
            else if(key == "down")
                {
                    obs->position[0] -= 0.1*obs->forward[0];
                    obs->position[1] -= 0.1*obs->forward[1];
                    obs->position[2] -= 0.1*obs->forward[2];

                    obs->look_at[0] -= 0.1*obs->forward[0];
                    obs->look_at[1] -= 0.1*obs->forward[1];
                    obs->look_at[2] -= 0.1*obs->forward[2];
                }
            else if(key == "left")
                {
                    obs->position[0] += 0.1*obs->left[0];
                    obs->position[1] += 0.1*obs->left[1];
                    obs->position[2] += 0.1*obs->left[2];

                    obs->look_at[0] += 0.1*obs->left[0];
                    obs->look_at[1] += 0.1*obs->left[1];
                    obs->look_at[2] += 0.1*obs->left[2];
                }
            else if(key == "right")
                {
                    obs->position[0] -= 0.1*obs->left[0];
                    obs->position[1] -= 0.1*obs->left[1];
                    obs->position[2] -= 0.1*obs->left[2];

                    obs->look_at[0] -= 0.1*obs->left[0];
                    obs->look_at[1] -= 0.1*obs->left[1];
                    obs->look_at[2] -= 0.1*obs->left[2];            
                }
            else if(key == "home")
                {
                    obs->position[0] += 0.1*obs->up[0];
                    obs->position[1] += 0.1*obs->up[1];
                    obs->position[2] += 0.1*obs->up[2];

                    obs->look_at[0] += 0.1*obs->up[0];
                    obs->look_at[1] += 0.1*obs->up[1];
                    obs->look_at[2] += 0.1*obs->up[2];            
                }
            else if(key == "end")
                {
                    obs->position[0] -= 0.1*obs->up[0];
                    obs->position[1] -= 0.1*obs->up[1];
                    obs->position[2] -= 0.1*obs->up[2];

                    obs->look_at[0] -= 0.1*obs->up[0];
                    obs->look_at[1] -= 0.1*obs->up[1];
                    obs->look_at[2] -= 0.1*obs->up[2];            
                }
            else if(key == "g")
                {
                    std::cout << "Viewport geometry : " 
                              << " [ " << geometry[0]
                              << " ; " << geometry[1]
                              << " ; " << geometry[2]
                              << " ; " << geometry[3]
                              << " ] " << std::endl;
                }
            // else do nothing
        }
    
}

//_________________________________________________________pointer_motion_event
void
Viewport::pointer_motion_event (int x, int y)
{
    if ((!visible) || (!has_focus))
        return;
    if (!child_has_focus) {
        dx = (x-(geometry[0]+geometry[2]/2.0))/geometry[2];
        dy = (y - (geometry[1]+geometry[3]/2.0))/geometry[3];
        //         Observer * obs = dynamic_cast< world::core::Observer *>(observer.get());
        //         obs->pointer_motion_event ((x-(geometry[0]+geometry[2]/2.0))/geometry[2], (y - (geometry[1]+geometry[3]/2.0))/geometry[3]);
    } else {
        for (unsigned int i=0; i<viewports.size(); i++) {
            if (viewports[i]->has_focus)
                viewports[i]->pointer_motion_event (x, y);
        }
    }
}

//_______________________________________________________________________render
void
Viewport::render ()
{
    if(button_pressed)
        {
            Observer * obs = dynamic_cast< world::core::Observer *>(observer.get());
            if(obs->allow_movement)
                {
                    obs->pointer_motion_event (dx, dy);
                }
        }
    glpython::core::Viewport::render();
}

void
Viewport::button_press_event (int button, int x, int y)
{
    glpython::core::Viewport::button_press_event(button,x,y);
    dx = (x-(geometry[0]+geometry[2]/2.0))/geometry[2];
    dy = (y - (geometry[1]+geometry[3]/2.0))/geometry[3];
    button_pressed = true;
}

void
Viewport::button_release_event (int button, int x, int y)
{
    glpython::core::Viewport::button_release_event(button,x,y);
    button_pressed = false;
}

//________________________________________________________________python_export
void
Viewport::python_export (void) {

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Viewport> >();

    class_<Viewport, bases<glpython::core::Viewport> > ("Viewport",
                                                        "======================================================================\n"
                                                        " A world::Viewport inherits core::Viewport and provides a method      \n"
                                                        " that permits to freefly in the scene                                 \n"
                                                        "======================================================================\n",
                                                        init< optional <tuple, tuple, bool, bool, std::string> > (
                                                                                                                  (arg("size") = make_tuple (1.0f,1.0f),
                                                                                                                   arg("position") = make_tuple (0.0f,0.0f),
                                                                                                                   arg("has_border") = false,
                                                                                                                   arg("is_ortho") = false,
                                                                                                                   arg("name") = "Viewport"),
                                                                                                                  "__init__ (size, position, has_border, name )\n"))
        
        .def ("key_press_event", &Viewport::key_press_event,
              "key_press_event (key)\n")

        .def ("button_press_event", &Viewport::button_press_event,
              "button_press_event (button, x, y)")
        
        .def ("button_release_event", &Viewport::button_release_event,
              "button_release_event (button,x,y)")
        
        
        .def ("pointer_motion_event", &Viewport::pointer_motion_event,
              "pointer_motion_event (x,y)\n")
        
        .def("save",&Viewport::save,"save(filename) : save a snapshot of the viewport in filename\n")
        ;       
}
