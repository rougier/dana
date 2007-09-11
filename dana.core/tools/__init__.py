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

import os

def CheckLibrary (env, libs = []):
    """ Check for libraries using pkg-config """
    for lib in libs:
        if env['VERBOSE']: print 'Checking for C library %s... ' % lib,
        if os.system ('pkg-config --exists %s' % lib) == 1:
            if env['VERBOSE']: print 'no'
            sys.exit(1)
        if env['VERBOSE']: print 'yes'
        env.ParseConfig ('pkg-config --cflags --libs %s' % lib)

def CheckModule (env, modules = []):
    """ Check for python modules """
    for module in modules:
        if env['VERBOSE']: print 'Checking for Python package %s... ' % module,
        try:
            __import__ (module)
        except:
            if env['VERBOSE']: print 'no'
            sys.exit(1)
        else:
            if env['VERBOSE']: print 'yes'

