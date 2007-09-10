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


def Modules (env, modules):
    """ Build a list of module """

    modules_target = []
    install_target = []
    for m in modules:
        name, libs = m
        module, install = Module (env, name, libs)
        if module:
            modules_target.extend (module)
        if install:
            install_target.extend (install)
    return modules_target, install_target


def Module (env, path, libs=[]):
    """ Make a python module """

    module_name = os.path.basename (path)
    library_name = path.replace ('/', '_')
    install = []

    # Library
    src = ['%s/%s' % (env["BUILDDIR"], s)
           for s in glob.glob ('%s/*.cc' % path) if '_export' not in s]
    if len(src):
        lib = env.SharedLibrary ('%s/lib%s' % (env["BUILDLIBDIR"], library_name),
                                 src,
                                 SHLIBPREFIX='',
                                 LIBS=env['LIBS'])
        install.extend (env.Install (env['LIBDIR'], lib))


    # Module
    src = ['%s/%s' % (env["BUILDDIR"], s)
           for s in glob.glob ('%s/*.cc' % path) if '_export' in s]
    if len(src):
        module = env.SharedLibrary ('%s/_%s' % (env["BUILDLIBDIR"], module_name),
                                    src,
                                    SHLIBPREFIX='',
                                    LIBPATH= env['BUILDLIBDIR'],
                                    LIBS = libs + env['LIBS'] + [library_name])
        env.Depends (module, lib)
        install.extend (env.Install (os.path.join (env['PYTHONDIR'], path), module))
    else:
        module = None

    src = ['%s/%s' % (env["BUILDDIR"], s) for s in glob.glob ('%s/*.py' % path)]
    install.extend (env.Install (os.path.join (env['PYTHONDIR'], path), src))

    # Headers
    headers = glob.glob ('%s/*.h' % path)
    install.extend (env.Install (os.path.join (env['INCLUDEDIR'], path), headers))

    env.Alias ('install', env['LIBDIR'])
    env.Alias ('install', os.path.join (env['PYTHONDIR'], path))
    env.Alias ('install', env['INCLUDEDIR'])

    return module, install

