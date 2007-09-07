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
import shutil
import fnmatch

def find (path, exclusions= []):
    """ Recursively find files not matching exclusion patterns """

    files = []
    for filename in os.listdir(path):
        excluded = False
        for pattern in exclusions:
            if fnmatch.fnmatch (filename, pattern):
                excluded = True
        if not excluded:
            fullname = os.path.join (path, filename)
            if os.path.isdir (fullname):
                files.append (fullname)
                files.extend (find (fullname, exclusions))
            else:
                files.append (fullname)
    return files

def make_archive_tree (path):
    """ Make a copy of current directory into path """

    exclusions = ['.*', '*.os', '*.pyc', '*.o', '*~',
                  '*.pdf', '*.ps', '*.so', '*.a',
                  '*.zip', '*.tgz', '*.gz', '*.bz2',
                  'manual', '.cache', '.build', 'dist', '.libs']
    exclusions.append (path)
    shutil.rmtree (path, ignore_errors=True)
    files = find ('.', exclusions)
    if not os.path.exists (path):
        os.mkdir (path)        

    files.sort()
    for f in files:
        if os.path.isdir (f):
            os.mkdir (os.path.join (path, f))
    for f in files:
        if not os.path.isdir (f):
            shutil.copy (f, os.path.join (path, f))

def clean_archive_tree (env, source, target):
    """ Remove path  """

    path = str(source[0])
    shutil.rmtree (path, ignore_errors=True)
    print "Done"


def make_dist_zip (env, name):
    """ Make a zip archive """
    
    env['ZIPFLAGS'] = ''
    env['ZIPSUFFIX'] = ''
    print "Making archive %s.zip" % name

    make_archive_tree (name)
    zip = env.Zip ('dist/%s.zip' % name, name)
    env.AlwaysBuild (zip)
    env.AddPostAction (zip, clean_archive_tree)
    env.Alias('dist-zip', zip)


def make_dist_tgz (env, name):
    """ Make a tgz archive """

    env["TARFLAGS"] = '--create --gzip'
    env["TARSUFFIX"] = ''
    print "Making archive %s.tgz" % name

    make_archive_tree (name)
    tgz = env.Tar ('dist/%s.tgz' % name, name)
    env.AlwaysBuild (tgz)
    env.AddPostAction (tgz, clean_archive_tree)
    env.Alias('dist-tgz', tgz)

def make_dist_bz2 (env, name):
    """ Make a tar.bz2 archive """

    env["TARFLAGS"] = '--create --bzip2'
    env["TARSUFFIX"] = ''
    print "Making archive %s.tgz" % name

    make_archive_tree (name)
    bz2 = env.Tar ('dist/%s.tar.bz2' % name, name)
    env.AlwaysBuild (bz2)
    env.AddPostAction (bz2, clean_archive_tree)
    env.Alias('dist-bz2', bz2)
