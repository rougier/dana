# DANA --- Distributed Asyncronous Numerical Adaptive computing library
# Copyright (c) 2006,2007,2008 Nicolas P. Rougier
#
# This file is part of DANA.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

PROJECT (DANA)
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
SET (CMAKE_INCLUDE_CURRENT_DIR TRUE)
SET (CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
SET (CMAKE_SKIP_RPATH:BOOL OFF)
SET (CMAKE_INSTALL_PREFIX /home/local/ CACHE PATH "Installation directory" FORCE)

# Check for libraries & packages
INCLUDE(FindPythonModule)
FIND_LIBRARY (DANA_CORE dana_core ${CMAKE_INSTALL_PREFIX}/lib)
IF (NOT DANA_CORE)
  message (FATAL_ERROR "dana core component not found, please install dana.core first")
ENDIF (NOT DANA_CORE)
FIND_PACKAGE       (Boost        REQUIRED)
FIND_PACKAGE       (LibXml2      REQUIRED)
FIND_PACKAGE       (PythonLibs   REQUIRED)
FIND_PACKAGE       (PythonInterp REQUIRED)
FIND_PACKAGE       (Numpy        REQUIRED)
FIND_PYTHON_MODULE (dana.core    REQUIRED)
FIND_PYTHON_MODULE (numpy        REQUIRED)
FIND_PYTHON_MODULE (OpenGL       REQUIRED)
FIND_PYTHON_MODULE (Image        REQUIRED)
FIND_PYTHON_MODULE (ImageFont    REQUIRED)
FIND_PYTHON_MODULE (cairo        REQUIRED)
FIND_PYTHON_MODULE (wx           REQUIRED)
INCLUDE_DIRECTORIES (${CMAKE_INSTALL_PREFIX}/include)
INCLUDE_DIRECTORIES (${NUMPY_INCLUDE_DIR})
INCLUDE_DIRECTORIES (${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES (${LIBXML2_INCLUDE_DIR})
INCLUDE_DIRECTORIES (${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES (${OPENGL_INCLUDE_DIR})
LINK_DIRECTORIES (${CMAKE_INSTALL_PREFIX}/lib)

# Guess where is our python site package
EXEC_PROGRAM ("${PYTHON_EXECUTABLE}"
             ARGS "-c 'import sys; print sys.version[:3]'"
             OUTPUT_VARIABLE PYTHON_VERSION)
SET(SITE_PACKAGE ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/dana)

ADD_SUBDIRECTORY (dana)
