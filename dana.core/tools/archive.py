#! /usr/bin/env python
#
# Copyright 2007 Nicolas Rougier
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

""" Archive builder for scons 

Example usage (from you Sconstruct file):
-----------------------------------------

from archive import Files, archive_builder

...
 
env.Append (BUILDERS = {'Archive' : archive_builder})

...

if 'dist-tgz' in COMMAND_LINE_TARGETS:
    archive = env.Archive ('archive.tgz',
                           Files ('.', include=['*'], exclude=['*~', '.*', '*.o']))
    env.Alias ('dist-tgz', archive)

"""

from SCons.Script import *
import os, os.path
import fnmatch
import tarfile, zipfile

# _________________________________________________________________________Files
def Files (path, include = ['*'],  exclude= []):
    """ Recursively find files in path matching include patterns list
        and not matching exclude patterns
    """

    files = []
    for filename in os.listdir (path):
        included = False
        excluded = False
        for pattern in include:
            if fnmatch.fnmatch (filename, pattern):
                included = True
                for pattern in exclude:
                    if fnmatch.fnmatch (filename, pattern):
                        excluded = True
                        break
                break
        if included and not excluded:
            fullname = os.path.join (path, filename)
            if os.path.isdir (fullname):
                files.extend (Files (fullname, include, exclude))
            else:
                files.append (fullname)
    return files

# _______________________________________________________________________Archive
def Archive (target, source, env):
    """ Make an archive from sources """

    path = os.path.basename (str(target[0]))
    type = os.path.splitext (path)[-1]
    if type == '.tgz' or type == '.gz':
        archive = tarfile.open (path, 'w:gz')
    elif type == '.bz2':
        archive = tarfile.open (path, 'w:bz2')
    elif type == '.zip':
        archive = zipfile.ZipFile (path, 'w')
        archive.add = archive.write
    else:
        print "Unknown archive type (%s)" % type
        return

    src = [str(s) for s in source if str(s) != path]
    for s in src:
        archive.add (s, os.path.join (os.path.basename (str(target[0])), s))
    archive.close()

# _________________________________________________________________ArchiveString
def ArchiveString (target, source, env):
    """ Information string for Archive """
    return 'Making archive %s' % os.path.basename (str (target[0]))

archive_builder = Builder (action = SCons.Action.Action(Archive, ArchiveString))


