PROJECT(dana)
SET(DANA_VERSION 1.0.0)

SET(CMAKE_INCLUDE_CURRENT_DIR TRUE)
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
SET(CMAKE_SKIP_RPATH:BOOL OFF)

# Setting default install path
SET (CMAKE_INSTALL_PREFIX /home/local/ CACHE PATH "Installation directory" FORCE)
MESSAGE (STATUS "Setting installation directory to ${CMAKE_INSTALL_PREFIX}" )

# Boost
FIND_PACKAGE (Boost REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

# XML
SET(LibXml2_FIND_QUIETLY TRUE)
FIND_PACKAGE (LibXml2 REQUIRED)
INCLUDE_DIRECTORIES (${LIBXML2_INCLUDE_DIR})


# Python
FIND_PACKAGE (PythonLibs REQUIRED)
INCLUDE_DIRECTORIES (${PYTHON_INCLUDE_PATH})


# Numpy
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

ADD_SUBDIRECTORY(dana)
