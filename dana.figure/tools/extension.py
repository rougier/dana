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

import glob
import sys
import os.path

# ________________________________________________________________ExtensionBuild
def ExtensionBuild (env, path, libs=[]):
    """ Build a python extension module from a given path """

    all = glob.glob (os.path.join (path, '*.cc'))
    module_name   = '_' + path.split('/')[-1]
    library_name  = path.replace('/', '_')

    # Make the library
    src = ['%s/%s' % (env["BUILDDIR"], s) for s in all if '_export' not in s]
    if len(src):
        fullname = os.path.join (env["BUILDDIR"], library_name)
        library = env.SharedLibrary (fullname, 
                                     src,
                                     CPPPATH= env['CPPPATH'],
                                     LIBPATH= [env['BUILDDIR']] + env['LIBPATH'],
                                     LIBS=env['LIBS'] + libs)
    else:
        library = None

    # Make the module
    env_mod = env.Copy()
    if sys.platform == 'darwin':
        env_mod['SHLINKFLAGS'] = '$LINKFLAGS -bundle -flat_namespace -undefined suppress'
        env_mod['SHLIBSUFFIX'] = '.so'

    srcs = ['%s/%s' % (env["BUILDDIR"], s) for s in all if '_export' in s]
    if len(srcs):
        fullname = os.path.join (env["BUILDDIR"], path, module_name)
        module = env_mod.SharedLibrary (fullname,
                                        srcs,
                                        SHLIBPREFIX='',
                                        CPPPATH= env['CPPPATH'],
                                        LIBPATH= [env['BUILDDIR']] + env['LIBPATH'],
                                        LIBS = env['LIBS'] + [library_name])
        env.Depends (module, library)
        env.Alias ('build', module)
    else:
        module = None
    return library, module

# ______________________________________________________________ExtensionInstall
def ExtensionInstall (env, path, library, module):
    """ Install a python extension module from a given path"""

    p = env.Install (os.path.join (env["PYTHONDIR"], path),
                     glob.glob (os.path.join(path, '*.py')))
    env.Alias ('install', p)
    h = env.Install (os.path.join (env["INCLUDEDIR"], path),
                     glob.glob (os.path.join(path, '*.h')))
    env.Alias ('install', h)

    l = env.Install (env['LIBDIR'], library)
    env.Alias ('install', env['LIBDIR'])

    m = env.Install (os.path.join (env['PYTHONDIR'], path), module)
    env.Alias ('install', os.path.join (env['PYTHONDIR'], path))

    return p + h + l + m, module

# _____________________________________________________________________Extension
def Extension (env, path, libs=[]):
    """ Build and install a python extension module from a given path"""

    l,m = ExtensionBuild (env, path, libs)
    targets, module = ExtensionInstall (env, path, l, m)
    files = []
    for target in targets:
        files.append ( [str(target), str(target.sources[0])] )
    return files, module


    

