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

import sys
import os.path
import distutils.sysconfig


def configure (env, conf, opts, filename):
    """ Configure """

    # Check for python library and header
    conf.env.Append (CPPPATH = [distutils.sysconfig.get_python_inc()])
    version = distutils.sysconfig.get_python_version()
    print 'Checking for C library python%s...' % version,
    if conf.CheckLibWithHeader ('python' + version, 'Python.h', 'c'):
        print 'yes'
    else:
        print 'no'
        sys.exit(1)
        
    # Check for boost python
    print 'Checking for C library boost_python...',
    if conf.CheckLibWithHeader ('boost_python-gcc-mt',
                                'boost/python.hpp', 'c++'):
        print 'yes'
        conf.env.Append (LIBS = 'boost_python-gcc-mt')
    else:
        if conf.CheckLibWithHeader ('boost_python-mt',
                                    'boost/python.hpp', 'c++'):
            print 'yes'
            conf.env.Append (LIBS = 'boost_python-mt')
        else:
            if conf.CheckLibWithHeader ('boost_python',
                                        'boost/python.hpp', 'c++'):            
                print 'yes'
                conf.env.Append (LIBS = 'boost_python')
            else:
                print 'no'
                sys.exit(1)

    # Check for boost thread
    print 'Checking for C library boost_thread...',
    if conf.CheckLibWithHeader ('boost_thread-gcc-mt',
                                'boost/thread.hpp', 'c++'):
        print 'yes'
        conf.env.Append (LIBS = 'boost_thread-gcc-mt')
    else:
        if conf.CheckLibWithHeader ('boost_thread-mt',
                                    'boost/thread.hpp', 'c++'):
            print 'yes'
            conf.env.Append (LIBS = 'boost_thread-mt')
        else:
            if conf.CheckLibWithHeader ('boost_thread',
                                        'boost/thread.hpp', 'c++'):            
                print 'yes'
                conf.env.Append (LIBS = 'boost_thread')
            else:
                print 'no'
                sys.exit(1)
        
    # Check for xml-2.0
    print 'Checking for C library xml-2.0...',
    if conf.CheckLib ('xml2'):
        print 'yes'
    else:
        print 'no'
        sys.exit(1)
    env.ParseConfig('pkg-config --cflags --libs libxml-2.0')

    # Check for numpy package
    print 'Checking for Python package numpy... ',
    try:
        import numpy
    except:
        print 'no'
        sys.exit(1)
    else:
        print 'yes'
        conf.env.Append (CPPPATH = numpy.get_include())

    # Save configuration
    opts.Update (env)
    opts.Save (filename, env)

