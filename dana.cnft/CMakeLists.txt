# DANA - Distributed Asyncronous Numerical Adaptive computing library
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

message ("=====================================================================")
message (" DANA - Distributed Asyncronous Numerical Adaptive computing library ")
message (" Copyright (c) 2006,2007,2008 Nicolas P. Rougier                     ")
message ("=====================================================================")

# The name of our project is "DANA". CMakeLists files in this project can
# refer to the root source directory of the project as ${DANA_SOURCE_DIR}
# and to the root binary directory of the project as ${DANA_BINARY_DIR}.
project (DANA)

# Some supplementary cmake scripts
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")


# CMAKE_INCLUDE_CURRENT_DIR is the same as:
#  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
#  for all targets.
SET(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# Gracefully handle rpath when installing
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
SET(CMAKE_SKIP_RPATH:BOOL OFF)

# Setting default install path
SET (CMAKE_INSTALL_PREFIX /home/local/ CACHE PATH "Installation directory" FORCE)
MESSAGE (STATUS "Setting installation directory to ${CMAKE_INSTALL_PREFIX}" )

# Check for mandatory stuff
FIND_LIBRARY (DANA_CORE dana_core ${CMAKE_INSTALL_PREFIX}/lib)
IF (NOT DANA_CORE)
  message (FATAL_ERROR "dana core component not found, please install dana.core first")
ENDIF (NOT DANA_CORE)
INCLUDE_DIRECTORIES (${CMAKE_INSTALL_PREFIX}/include)
LINK_DIRECTORIES (${CMAKE_INSTALL_PREFIX}/lib)
FIND_PACKAGE (Boost REQUIRED)
FIND_PACKAGE (LibXml2 REQUIRED)
FIND_PACKAGE (PythonLibs REQUIRED)
FIND_PACKAGE (PythonInterp REQUIRED)
FIND_PACKAGE (Numpy REQUIRED)
INCLUDE_DIRECTORIES (${NUMPY_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
INCLUDE_DIRECTORIES (${LIBXML2_INCLUDE_DIR})
INCLUDE_DIRECTORIES (${PYTHON_INCLUDE_PATH})


# Guess python site package
EXEC_PROGRAM ("${PYTHON_EXECUTABLE}"
             ARGS "-c 'import sys; print sys.version[:3]'"
             OUTPUT_VARIABLE PYTHON_VERSION)
SET(SITE_PACKAGE ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/dana)

# Recurse into the "dana" subdirectory. This does not actually
# cause another cmake executable to run. The same process will walk
# through the project's entire directory structure.
add_subdirectory (dana)
