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

# The name of our project is "DANA". CMakeLists files in this project can
# refer to the root source directory of the project as ${DANA_SOURCE_DIR}
# and to the root binary directory of the project as ${DANA_BINARY_DIR}.
project (DANA)
set     (DANA_VERSION 1.0.0)

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


# Try fo find boost library
FIND_PACKAGE (Boost REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

# Try fo find XML library
SET(LibXml2_FIND_QUIETLY TRUE)
FIND_PACKAGE (LibXml2 REQUIRED)
INCLUDE_DIRECTORIES (${LIBXML2_INCLUDE_DIR})

# Try to find python library
FIND_PACKAGE (PythonLibs REQUIRED)
INCLUDE_DIRECTORIES (${PYTHON_INCLUDE_PATH})

# Try to check for numpy package
FIND_PACKAGE (PythonInterp REQUIRED)
EXEC_PROGRAM ("${PYTHON_EXECUTABLE}"
             ARGS "-c 'import numpy; print numpy.get_include()'"
             OUTPUT_VARIABLE NUMPY_INCLUDE_PATH
             RETURN_VALUE NUMPY_NOT_FOUND)
INCLUDE_DIRECTORIES (${NUMPY_INCLUDE_PATH})

# Tell what is missing
IF (NOT Boost_FOUND)
  message ("No boost python library found (http://www.boost.org)")
ENDIF (NOT Boost_FOUND)
IF (NOT LIBXML2_FOUND)
  message ("No xml2 library found (http://xmlsoft.org) ")
ENDIF (NOT LIBXML2_FOUND)
IF (NUMPY_NOT_FOUND)
  message ("No numpy package found  (http://numpy.scipy.org)")
ENDIF (NUMPY_NOT_FOUND)
IF (NOT PYTHONINTERP_FOUND)
  message ("No python interpreter found (http://www.python.org)")
ENDIF (NOT PYTHONINTERP_FOUND)

# Exit if something required is missing
IF (NOT Boost_FOUND OR NOT LIBXML2_FOUND OR NUMPY_NOT_FOUND OR NOT PYTHONINTERP_FOUND)
  message (FATAL_ERROR "Some library or package are missing, please consider installing them.")
ENDIF (NOT Boost_FOUND OR NOT LIBXML2_FOUND OR NUMPY_NOT_FOUND OR NOT PYTHONINTERP_FOUND)


# Guess python site package
EXEC_PROGRAM ("${PYTHON_EXECUTABLE}"
             ARGS "-c 'import sys; print sys.version[:3]'"
             OUTPUT_VARIABLE PYTHON_VERSION)
SET(SITE_PACKAGE ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages/dana)

# Recurse into the "dana" subdirectory. This does not actually
# cause another cmake executable to run. The same process will walk
# through the project's entire directory structure.
add_subdirectory (dana)
