//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <string>
#include <fstream>
#include <iostream>
#include "model.h"

using namespace boost::python;
using namespace glpython::objects;


//________________________________________________________________________Model
Model::Model (std::string filename,
              std::string name) : core::Object (name)

{

    file = lib3ds_file_load(filename.c_str());
    if (!file) {
        std::cerr << "Error loading file '" << filename << "'\n";
        return;
    }
    if (!file->nodes) {
        Lib3dsMesh *mesh;
        Lib3dsNode *node;

        for(mesh = file->meshes; mesh != NULL; mesh = mesh->next) {
            node = lib3ds_node_new_object();
            strcpy (node->name, mesh->name);
            node->parent_id = LIB3DS_NO_PARENT;
            lib3ds_file_insert_node(file, node);
        }
    }
    lib3ds_file_eval (file, 1.0f);
}

//_______________________________________________________________________~Model
Model::~Model (void)
{}

//__________________________________________________________________render_node
void
Model::render_node (Lib3dsNode *node)
{
    Lib3dsNode *p;
    for (p=node->childs; p!=0; p=p->next)
        render_node(p);

    if (node->type != LIB3DS_OBJECT_NODE)
        return;

    if (strcmp(node->name,"$$$DUMMY")==0)
        return;
    
    Lib3dsMesh *mesh;

    mesh = lib3ds_file_mesh_by_name(file, node->data.object.morph);
    if (mesh == NULL)
        mesh = lib3ds_file_mesh_by_name(file, node->name);

    if (!mesh->user.d) {
        ASSERT(mesh);
        if (!mesh) {
            return;
        }

        mesh->user.d=glGenLists(1);
        glNewList (mesh->user.d, GL_COMPILE);
        unsigned p;
        Lib3dsVector *normalL = (Lib3dsVector *) malloc(3*sizeof(Lib3dsVector)*mesh->faces);
        Lib3dsMaterial *oldmat = (Lib3dsMaterial *)-1;
        Lib3dsMatrix M;
        lib3ds_matrix_copy(M, mesh->matrix);
        lib3ds_matrix_inv(M);
        glMultMatrixf(&M[0][0]);
        lib3ds_mesh_calculate_normals(mesh, normalL);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glDisable (GL_DEPTH_TEST);
        glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        glPolygonOffset (1.0f, 1.0f);
        glEnable (GL_POLYGON_OFFSET_FILL);

        for (p=0; p<mesh->faces; ++p) {
            Lib3dsFace *f=&mesh->faceL[p];
            Lib3dsMaterial *mat=0;
            if (f->material[0]) {
                mat=lib3ds_file_material_by_name(file, f->material);
            }

            if (mat != oldmat) {
                if (mat) {
//                    if( mat->two_sided )
//                        glDisable(GL_CULL_FACE);
//                    else
//                        glEnable(GL_CULL_FACE);
                    glMaterialfv(GL_FRONT, GL_AMBIENT, mat->ambient);
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat->diffuse);
                    glMaterialfv(GL_FRONT, GL_SPECULAR, mat->specular);
                    glMaterialf(GL_FRONT, GL_SHININESS, pow (2, 10.0*mat->shininess));
                    glColor4f (mat->diffuse[0],mat->diffuse[1], mat->diffuse[2], .5);
                } else {
                    static const Lib3dsRgba a={0.7, 0.7, 0.7, 1.0};
                    static const Lib3dsRgba d={0.7, 0.7, 0.7, 1.0};
                    static const Lib3dsRgba s={1.0, 1.0, 1.0, 1.0};
                    glMaterialfv(GL_FRONT, GL_AMBIENT, a);
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, d);
                    glMaterialfv(GL_FRONT, GL_SPECULAR, s);
                    glMaterialf(GL_FRONT, GL_SHININESS, pow(2, 10.0*0.5));
                }
                oldmat = mat;
            }
            glBegin(GL_TRIANGLES);
            glNormal3fv(f->normal);
            for (int i=0; i<3; ++i) {
                glNormal3fv(normalL[3*p+i]);
                glVertex3fv(mesh->pointL[f->points[i]].pos);
            }
            glEnd();
        }
        glDisable (GL_POLYGON_OFFSET_FILL);
        glColor4f (1,1,1,.1);
        glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        for (p=0; p<mesh->faces; ++p) {
            Lib3dsFace *f=&mesh->faceL[p];
            glBegin(GL_TRIANGLES);
            glNormal3fv(f->normal);
            for (int i=0; i<3; ++i)
                glVertex3fv(mesh->pointL[f->points[i]].pos);
            glEnd();
        }
        
        free(normalL);
        glEndList();
    }

    if (mesh->user.d) {
        Lib3dsObjectData *d;
        glPushMatrix();
        d=&node->data.object;
        glMultMatrixf(&node->matrix[0][0]);
        glTranslatef(-d->pivot[0], -d->pivot[1], -d->pivot[2]);
        glCallList(mesh->user.d);
        glPopMatrix();
    }
}



//_______________________________________________________________________render
void
Model::render (void)
{
    Lib3dsNode *p;
    for (p=file->nodes; p!=0; p=p->next)
      render_node(p);
}

//________________________________________________________________python_export
void
Model::python_export (void)
{

    using namespace boost::python;
    register_ptr_to_python< boost::shared_ptr<Model> >();

    class_<Model, bases< core::Object> > ("Model",
    " ______________________________________________________________________\n"
    "                                                                       \n"
    " ______________________________________________________________________\n",
    init<std::string, optional <std::string> > (
        (arg("filename"),
         arg("name") = "Model"),
        "__init__ (filename, name)\n"))
    ;
}
