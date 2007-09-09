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

import os, os.path
import glob

def Module (env, path, libs=[]):
    """ Make a python module """

    env.BuildDir (env["BUILDDIR"], '.', duplicate=0)
    modname         = os.path.basename (path)
    libname         = path.replace ('/', '_')
    lib_install_dir = '%s/lib' % env['PREFIX']
    lib_include_dir = '%s/include' % env['PREFIX']
    mod_install_dir = os.path.join (
        '%s/lib/%s/site-packages' % (env['PREFIX'], env['PYTHON']), path)

    # Library
    src = ['%s/%s' % (env["BUILDDIR"], s)
           for s in glob.glob ('%s/*.cc' % path) if '_export' not in s]
    if len(src):
        lib = env.SharedLibrary ('%s/lib%s' % (env["LIBDIR"], libname),
                                 src,
                                 SHLIBPREFIX='',
                                 LIBS=env['LIBS'])
        env.Install (lib_install_dir, lib)
        env.Alias('install', lib_install_dir)

    # Module
    src = ['%s/%s' % (env["BUILDDIR"], s)
           for s in glob.glob ('%s/*.cc' % path) if '_export' in s]
    if len(src):
        module = env.SharedLibrary ('%s/_%s' % (env["LIBDIR"], modname),
                                    src,
                                    SHLIBPREFIX='',
                                    LIBPATH= env['LIBDIR'],
                                    LIBS = libs + env['LIBS'] + [libname])
        env.Depends (module, lib)
        env.Install (mod_install_dir, module)
    else:
        module = None
    src = ['%s/%s' % (env["BUILDDIR"], s) for s in glob.glob ('%s/*.py' % path)]
    env.Install (mod_install_dir, src)
    env.Alias ('install', mod_install_dir)

    headers = glob.glob ('%s/*.h' % path)
    env.Install (os.path.join (lib_include_dir, path), headers)
    env.Alias ('install', lib_include_dir)

    return module

