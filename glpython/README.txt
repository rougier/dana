
GLPython, an interactive OpenGL python perminal 
Copyright (c) 2006-2007 Nicolas Rougier.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.
-------------------------------------------------------------------------------


Description
-------------------------------------------------------------------------------
GLPython is (supposed to be) a multiplatform, free software project that offers
an OpenGL oriented python shell designed for efficient interactive work with
all major GUI toolkits in a non-blocking manner as well as a library to build
customized OpenGL objects using python as the basic language.


Overview
-------------------------------------------------------------------------------

GLPython is basically a Python (or IPython) terminal that is emulated inside an
OpenGL window. It supports completion, history, ANSI codes as well as some GNU
readline features. Usual key bindings are available as well as scrolling
through the mouse scroll button.

GLPython relies on backends such as GTK, WX, SDL or Qt that are able to open an
OpenGL context and handle keyboard and mouse events when necessary. In this
sense, GLUT is not suitable since we do not have control over the event loop
(we can only enter the main event loop once and for all). You're able to choose
the backend to be used when you start glpython using the '-b' option.

From a technical point of view, the terminal area is rendered inside a
framebuffer object that is then displayed as a texture when necessary. This
ensures some decent speed when displaying the terminal. Python standard inputs
and outputs are redirected to the terminal (this is the reason why we need to
have some control over the main event loop) while "C/C++" standard inputs and
outputs are not (I did not find yet a simple solution to redirect standard I/O
from "C/C++").

The size and position of the terminal is pretty flexible: it can takes the
entire screen or be reduced to a relative or absolute size anywhere on the
window. Key and mouse events are only processed when mouse is inside the
terminal area.

GLPython package also defines an "Object" class that is the base class for
displaying objects inside the window. Since I'm not sure how to handle both
OpenGL and Global Interpreter Lock (GIL) from Python, the solution I retained
was to refresh OpengGL scene at fixed time intervals. Hence, you cannot
directly issued OpenGL command from inside the terminal or you'll certainly get
some unexpected effects. However, if you define some object with a "render"
method and append this object to the main viewport ("view"), then it will be
displayed immediately (hopefully).

