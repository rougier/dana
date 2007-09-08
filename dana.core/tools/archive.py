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
    archive = env.Archive ('archive',
                           Files ('.', include=['*'], exclude=['*~', '.*', '*.o']),
                           ARCHIVE_TYPE = 'tgz')
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

    if not env.has_key ('ARCHIVE_TYPE'):
        env["ARCHIVE_TYPE"] = 'tgz' 
    
    if env["ARCHIVE_TYPE"] == 'tgz':
        path = '%s.tar.gz' % os.path.basename (str(target[0]))
        archive = tarfile.open (path, 'w:gz')
    elif env["ARCHIVE_TYPE"] == 'bz2':
        path = '%s.tar.bz2' % os.path.basename (str(target[0]))
        archive = tarfile.open (path, 'w:bz2')
    elif env["ARCHIVE_TYPE"] == 'zip':
        path = '%s.zip' % os.path.basename (str(target[0]))
        archive = tarfile.open (path, 'w:bz2')
    else:
        print "Unknown archive type (%s)" % env["ARCHIVE_TYPE"]
        print "Known types are: 'tgz', 'bz2' and 'zip'"
        return

    src = [str(s) for s in source if str(s) != path]
    for s in src:
        archive.add (s, os.path.join (os.path.basename (str(target[0])), s))
    archive.close()
    print 'Done !'

# _________________________________________________________________ArchiveString
def ArchiveString (target, source, env):
    """ Information string for Archive """

    if not env.has_key ('ARCHIVE_TYPE'):
        env["ARCHIVE_TYPE"] = 'tgz'
    path = os.path.basename (str (target[0]))
    if env["ARCHIVE_TYPE"] == 'tgz':
        return 'Making archive %s.tar.gz' % path
    elif env["ARCHIVE_TYPE"] == 'bz2':
        return 'Making archive %s.tar.bz2' % path
    elif env["ARCHIVE_TYPE"] == 'zip':
        return 'Making archive %s.tar.zip' % path
    else:
        return 'Making archive %s' % path


archive_builder = Builder (action = SCons.Action.Action(Archive, ArchiveString))


