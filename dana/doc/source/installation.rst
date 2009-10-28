.. title:: Installation
.. include:: common.rst

Standard installation                                                          
===============================================================================

DANA  is a python  package that  may be  used without  installation and  be ran
in-place, provided  you have a running  `numpy`_ installation.  If  you want to
experiment  with  DANA and  run  the  examples before  to  install  it on  your
development machine, add either the  extracted dana source archive directory or
the compressed runtime egg to your ``PYTHONPATH``.

On Windows you can specify this from a command line:

.. code-block:: none

   set PYTHONPATH c:\path\to\dana\;%PYTHONPATH%

On Mac OS X, Linux or on Windows under cygwin using bash:

.. code-block:: none

   set PYTHONPATH /path/to/dana/:$PYTHONPATH
   export PYTHONPATH

If you have  downloaded a runtime egg instead of the  source archive, you would
need to specify the filename of the egg in place of `dana`.

To  make  DANA  available  to  all  users,  or  to  avoid  having  to  set  the
``PYTHONPATH``  for  each  session,  you  can install  it  into  your  python's
site-packages  directory. From  a command  prompt on  Windows, change  into the
extracted DANA source archive directory and type:

.. code-block:: none

   python setup.py install

On Mac OS X  and Linux you will need to do the above  as a priveleged user; for
example using sudo:

.. code-block:: none

   sudo python setup.py install

Once installed  you should be able  to import dana from  any terminal without
setting the ``PYTHONPATH``.  If you have setuptools installed,  you can install
or upgrade to the latest version of dana using ``easy_install``:

.. code-block:: none

   easy_install -U dana

On Mac OS X  and Linux you may need to run the above  as a priveleged user; for
example:

.. code-block:: none

   sudo easy_install -U dana

Subversion repository                                                          
===============================================================================

The subversion repository houses the source trunk, documentation and examples.
You can browse the repository or check out a working copy using an SVN client:

.. code-block:: none

   svn checkout svn://scm.gforge.inria.fr/svn/dana/trunk/dana dana-svn
