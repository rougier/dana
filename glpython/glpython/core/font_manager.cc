//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#include <iostream>
#include <fontconfig/fontconfig.h>
#include <FTGL/FTGLPolygonFont.h>
#include <FTGL/FTGLOutlineFont.h>
#include <FTGL/FTGLTextureFont.h>
#include <FTGL/FTGLPixmapFont.h>
#include <FTGL/FTGLBitmapFont.h>
#include "font_manager.h"

using namespace glpython::core;

//_________________________________________________________________________Font
Font::Font (std::string name, std::string type, int size)
{ 
    std::string filename = "/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf";

    FcFontSet * fs;
    FcPattern * pat;
    FcResult    result;
    if (!FcInit ()) {
    	fprintf (stderr, "Can't init font config library\n");
    } else {
        pat = FcNameParse ((FcChar8 *) name.c_str());
        if (pat) {
            FcConfigSubstitute (0, pat, FcMatchPattern);
            FcDefaultSubstitute (pat);
            FcPattern * match;
            fs = FcFontSetCreate ();
            match = FcFontMatch (0, pat, &result);
            if (match)
                FcFontSetAdd (fs, match);
            FcPatternDestroy (pat);
            if (fs) {
                FcChar8	*file;
                FcPatternGetString (fs->fonts[0], FC_FILE, 0, &file);
                filename = std::string ((char *)file);
                FcFontSetDestroy (fs);
            }
        }
        // FcFini ();
    }

    if (type == "outline") {
        font = new FTGLOutlineFont (filename.c_str());
    } else if (type == "polygon") {
        font = new FTGLPolygonFont (filename.c_str());
    } else if (type == "bitmap") {
        font = new FTGLBitmapFont (filename.c_str());
    } else if (type == "pixmap") {
        font = new FTGLPixmapFont (filename.c_str());
    } else {
        font = new FTGLTextureFont (filename.c_str());
    }
    if ((font->Error()) || (!font->FaceSize (size))) {
        std::cerr << "Failed to open font '" << filename << "'" << std::endl;
        font = 0;
    } else {
        this->name = name;
        this->type = type;
        this->size = size;
    }
}

//________________________________________________________________________~Font
Font::~Font (void)
{}

//________________________________________________________________________fonts
std::vector<Font *> FontManager::fonts;

//__________________________________________________________________FontManager
FontManager::FontManager (void) 
{}

//_________________________________________________________________~FontManager
FontManager::~FontManager (void)
{}

//__________________________________________________________________________get
FTFont *
FontManager::get (std::string name, std::string type, int size)
{
    for (unsigned int i=0; i<fonts.size(); i++)
        if ((fonts[i]->name == name) && (fonts[i]->type == type) && (fonts[i]->size == size))
            return fonts[i]->font;

    Font *font = new Font (name, type, size);
    if (font->font) {
        fonts.push_back (font);
        return font->font;
    }
    delete font;
    return 0;
}

