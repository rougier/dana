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

import os, shutil, sys
Import('env') # exported by parent SConstruct


svn_version = os.popen('svnversion .').read()[:-1]
svn_version = svn_version.split(':')[-1]

DEBNAME = 'dana.core'
DEBVERSION = "svn"
DEBMAINT = "Nicolas P.Rougier [Nicolas.Rougier@loria.fr]"
DEBARCH = "x86_64"
DEBDEPENDS = "python2.5-minimal, libboost-python1.33.1"
DEBDESC = "DANA core"

DEBFILES = [
    # Now we specify the files to be included in the .deb
    # Where they should go, and where they should be copied from.
    # If you have a lot of files, you may wish to generate this 
    # list in some other way.
    ("usr/lib/libdana_core.so",       "#/.libs/libdana_core.so"),
    ]

# This is the debian package we're going to create
debpkg = '#%s_%s-%s_%s.deb' % (DEBNAME, DEBVERSION, svn_version, DEBARCH)

# and we want it to be built when we build 'dist-deb'
env.Alias("dist-deb", debpkg)

DEBCONTROLFILE = os.path.join(DEBNAME, "DEBIAN/control")

# This copies the necessary files into place into place.
# Fortunately, SCons creates the necessary directories for us.
for f in DEBFILES:
    # We put things in a directory named after the package
    dest = os.path.join(DEBNAME, f[0])
    # The .deb package will depend on this file
    env.Depends(debpkg, dest)
    # Copy from the the source tree.
    env.Command(dest, f[1], Copy('$TARGET','$SOURCE'))
    # The control file also depends on each source because we'd like
    # to know the total installed size of the package
    env.Depends(DEBCONTROLFILE, dest)

# Now to create the control file:

CONTROL_TEMPLATE = """
Package: %s
Priority: extra
Section: misc
Installed-Size: %s
Maintainer: %s
Architecture: %s
Version: %s-%s
Depends: %s
Description: %s

"""
env.Depends(debpkg,DEBCONTROLFILE )

# The control file should be updated when the SVN version changes
env.Depends(DEBCONTROLFILE, env.Value(svn_version))

# This function creates the control file from the template and info
# specified above, and works out the final size of the package.
def make_control(target=None, source=None, env=None):
    installed_size = 0
    for i in DEBFILES:
        installed_size += os.stat(str(env.File(i[1])))[6]
        control_info = CONTROL_TEMPLATE % (
            DEBNAME, installed_size, DEBMAINT, DEBARCH, DEBVERSION,
            svn_version, DEBDEPENDS, DEBDESC)
    f = open(str(target[0]), 'w')
    f.write(control_info)
    f.close()
    
# We can generate the control file by calling make_control
env.Command(DEBCONTROLFILE, None, make_control)

# And we can generate the .deb file by calling dpkg-deb
env.Command(debpkg, DEBCONTROLFILE,
            "dpkg-deb -b %s %s" % ("deb/%s" % DEBNAME, "$TARGET"))
