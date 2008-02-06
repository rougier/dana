# A python module finder for CMake
#
# Usage:
#    INCLUDE (FindPythonModule)
#    FIND_PYTHON_MODULE (<MODULE> [<MODULE>]* [REQUIRED])
#
# When the 'REQUIRED' argument is set, macros will fail with an error
# when module(s) cannot be found
#
# It sets the following variables for each module xyz:
#   python_xyz_FOUND      ... set to true if module(s) exist
#                             set to false else
#
# Examples:
#   FIND_PYTHON_MODULE (gtk gdk xyz)
#   defines: python_gtk_FOUND (true)
#            python_gdk_FOUND (true)
#            python_xyz_FOUND (false)
#
# Copyright (c) 2008 Nicolas Rougier <Nicolas.Rougier@loria.fr>
#
# Redistribution and use, with or without modification, are permitted
# provided that the following conditions are met:
# 
#    1. Redistributions must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#    2. The name of the author may not be used to endorse or promote
#       products derived from this software without specific prior
#       written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


macro (FIND_PYTHON_MODULE module0)
    FIND_PACKAGE (PythonInterp REQUIRED)
    set (is_required false)

    # Search for a 'REQUIRED' module
    foreach (module ${module0} ${ARGN})
      if (module STREQUAL "REQUIRED")
        set (is_required true)
      endif (module STREQUAL "REQUIRED")
    endforeach (module ${module0} ${ARGN})

    # Check for all modules if not cached
    foreach (module ${module0} ${ARGN})
      # if module is "REQUIRED", it is not a module
      if (module STREQUAL "REQUIRED")
        set (is_required true)
      else (module STREQUAL "REQUIRED")
        # Checks for specific module if not cached
        if (NOT python_${module}_FOUND)
          # module is supposed to be present by default
          set (python_${module}_FOUND true CACHE INTERNAL "")
          
          # Check whether module can be imported or not
          exec_program ("${PYTHON_EXECUTABLE}"
            ARGS "-c 'import ${module}'"
            OUTPUT_VARIABLE ${module}_OUTPUT
            RETURN_VALUE ${module}_STATUS)
          # Process result
          if (${module}_STATUS)
            if (is_required)
              message (FATAL_ERROR "Python module ${module} missing")
            else (is_required)
              message (SEND_ERROR "Python module ${module} missing")
            endif (is_required)
            set (python_${module}_FOUND false)
          else (${module}_STATUS)
            message (STATUS "Python module ${module} found")
          endif (${module}_STATUS)
        endif (NOT python_${module}_FOUND)
      endif (module STREQUAL "REQUIRED")
    endforeach (module ${ARGN})
endmacro (FIND_PYTHON_MODULE)
