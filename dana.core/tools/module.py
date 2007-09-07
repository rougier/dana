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

def make_module (env, conf, path, libs=[]):
    """ Make a python module """

    build_dir       = env["CACHEDIR"]
    modname         = os.path.basename (path)
    libname         = path.replace ('/', '_')
    lib_install_dir = env['LIBDIR']
    lib_include_dir = env['INCLUDEDIR']
    mod_install_dir = os.path.join (env['PYTHONDIR'], path)

    # Library
    src = ['%s/%s' % (build_dir, s)
           for s in glob.glob ('%s/*.cc' % path) if '_export' not in s]
    if len(src):
        lib = env.SharedLibrary ('.libs/lib%s' % libname,
                                 src,
                                 SHLIBPREFIX='',
                                 LIBS=env['LIBS'])
        env.Install (lib_install_dir, lib)
        env.Alias('install', lib_install_dir)

    # Module
    src = ['%s/%s' % (build_dir, s)
           for s in glob.glob ('%s/*.cc' % path) if '_export' in s]
    if len(src):
        module = env.SharedLibrary ('.libs/_%s' % modname,
                                    src,
                                    SHLIBPREFIX='',
                                    LIBPATH= ['.libs'],
                                    LIBS = libs + env['LIBS'] + [libname])
        env.Install (mod_install_dir, module)

    src = ['%s/%s' % (build_dir, s) for s in glob.glob ('%s/*.py' % path)]
    env.Install (mod_install_dir, src)
    env.Alias ('install', mod_install_dir)

    headers = glob.glob ('%s/*.h' % path)
    env.Install (os.path.join (lib_include_dir, path), headers)
    env.Alias ('install', lib_include_dir)

