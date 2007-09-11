//
// Copyright (C) 2007 Nicolas Rougier
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// $Id$

#ifndef __GLPYTHON_CORE_FONT_MANAGER_H__
#define __GLPYTHON_CORE_FONT_MANAGER_H__

#include <string>
#include <vector>
#include <FTGL/FTFont.h>


namespace glpython { namespace core {

    class Font {
        public:
            std::string name;
            std::string type;
            int         size;
            FTFont *    font;
        public:
            Font (std::string name = "bitstream vera sans",
                  std::string type = "texture",
                  int size = 12);
            ~Font (void);
    };

    class FontManager {
        public:
            static std::vector<Font *> fonts;
    
        public:
            FontManager (void);
            ~FontManager (void);
            static FTFont *get (std::string name = "bitstream vera sans",
                                std::string type = "texture",
                                int size = 12);
    };
}}

#endif
