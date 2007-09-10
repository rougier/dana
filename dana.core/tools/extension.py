#! /usr/bin/env python
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Extension builder for scons 

Example usage (from you Sconstruct file):
-----------------------------------------

from extension import extension_builder

...
 
env.Append (BUILDERS = {'Extension' : extension_builder})

...

env.Extension ('core', 'core.cc')

"""

from SCons.Script import *

# _____________________________________________________________________Extension
def Extension (target, source, env):

    extension_dirname  = os.path.dirname (str(target[0]))
    extension_basename = os.path.basename (str(target[0]))

    # Make the library
    srcs = ['%s/%s' % (env["BUILDDIR"], str(s))
            for s in source if '_export' not in str(s)]
    if len(srcs):
        library_name = os.path.join(extension_dirname, extension_basename)
        library = env.SharedLibrary (library_name,
                                     srcs,
                                     LIBPATH= env['BUILDDIR'],
                                     LIBS=env['LIBS'])
        env.Install (env['LIBDIR'], library)
        env.Alias ('install', env['LIBDIR'])

    # Make the module
    srcs = ['%s/%s' % (env["BUILDDIR"], str(s))
            for s in source if '_export' in str(s)]
    if len(srcs):
        path = ''
        for p in extension_basename.split('_'):
            path = os.path.join (path, p)
        module_name = os.path.join (extension_dirname, path,
                                    '_' + extension_basename.split('_')[-1])
        module = env.SharedLibrary (module_name,
                                    srcs,
                                    SHLIBPREFIX='',
                                    LIBPATH= env['BUILDDIR'],
                                    LIBS = env['LIBS'] + [extension_basename])
        env.Depends (module, library)
        env.Install (os.path.join (env['LIBDIR'], path), module)
        env.Alias ('install', os.path.join (env['PYTHONDIR'], path))

    return 0


# _______________________________________________________________ExtensionString
def ExtensionString (target, source, env):
    """ Information string for Archive """

    extension_basename = os.path.basename (str(target[0]))
    return 'Making extension %s' % (extension_basename.split('_')[-1])

extension_builder = Builder (
    action = SCons.Action.Action(Extension, ExtensionString))

