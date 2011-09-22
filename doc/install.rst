===============================================================================
Installation                                                                   
===============================================================================

.. only:: html

   .. contents::
      :local:
      :depth: 1

Dana is a python package and depends  on a number of python components that can
be easily installed (see instructions on respective homepages):

* **numpy**, http://numpy.scipy.org

  NumPy is the fundamental package for  scientific computing in Python. It is a
  Python library that provides a multidimensional array object, various derived
  objects (such as  masked arrays and matrices), and  an assortment of routines
  for  fast  operations  on  arrays,  including  mathematical,  logical,  shape
  manipulation,  sorting, selecting,  I/O, discrete  Fourier  transforms, basic
  linear  algebra, basic  statistical  operations, random  simulation and  much
  more.

* **scipy**, http://www.scipy.org

  SciPy is  a collection of  mathematical algorithms and  convenience functions
  built on  the Numpy extension  for Python. It  adds significant power  to the
  interactive Python  session by exposing  the user to high-level  commands and
  classes  for the  manipulation  and  visualization of  data.  With SciPy,  an
  interactive Python  session becomes a  data-processing and system-prototyping
  environment rivaling sytems such as Matlab, IDL, Octave, R-Lab, and SciLab.


----

Optionaly, here are also some components and tools you might consider installing:

* **matplotlib**, http://matplotlib.sourceforge.net

  matplotlib is a python 2D plotting library which produces publication quality
  figures in a variety of  hardcopy formats and interactive environments across
  platforms. matplotlib can  be used in python scripts,  the python and ipython
  shell (ala matlab or mathematica), web application servers, and six graphical
  user interface toolkits.

* **IPython**, http://ipython.scipy.org/moin/FrontPage

  The goal of IPython is  to create a comprehensive environment for interactive
  and  exploratory computing.  To  support,  this goal,  IPython  has two  main
  components:  An  enhanced  interactive  Python shell.   An  architecture  for
  interactive  parallel computing.   All of  IPython is  open  source (released
  under the revised  BSD license). You can see what  projects are using IPython
  here, or check out the talks and presentations we have given about IPython.



Using DANA in-place                                                            
===============================================================================
Dana can be ran in-place without any installation and thus you might want to
experiment first with dana before you install it on your development
machine. To do this, add either the extracted dana source archive directory or
the compressed runtime egg to your ``PYTHONPATH``.

On Windows you can specify this from a command line::

   set PYTHONPATH c:\path\to\dana\;%PYTHONPATH%

On Mac OS X, Linux or on Windows under cygwin using bash::

   export PYTHONPATH=/path/to/dana/:$PYTHONPATH

or, using tcsh (or a variant)::

   setenv PYTHONPATH /path/to/dana/:$PYTHONPATH

If you have downloaded a runtime egg instead of the source archive, you would
specify the filename of the egg in place of ``dana/``.



Quick installation                                                             
===============================================================================
If you have *setuptools* installed, you can install or upgrade to the latest
version of dana using ``easy_install``::

   easy_install -U dana

On Mac OS X and Linux you may need to run the above as a priveleged user; for
example::

   sudo easy_install -U dana



Manual installation                                                            
===============================================================================
To avoid having to set the ``PYTHONPATH`` for each session, you can install
DANA into your personal Python's site-packages directory if you have one. If
you do not have one yet, you can create it anywhere on your disk and add this
directory to your ``PYTHONPATH``.

From a command prompt on Windows, change into the extracted dana source
archive directory and type::

  python setup.py install --prefix=C:\path\to\local\site-packages\;

On Mac OS X and Linux you will need to do the same::

  sudo python setup.py install --prefix=/path/to/local/site-packages

Once installed you should be able to import dana from any terminal without
setting the ``PYTHONPATH``.

----

To make dana available to all users you can install it into your Python's
site-packages directory.

From a command prompt on Windows, change into the extracted dana source
archive directory and type::

   python setup.py install

On Mac OS X and Linux you will need to do the above as a priveleged user; for
example using sudo::

   sudo python setup.py install



Testing installation                                                           
===============================================================================

Once you've installed DANA, it is very important you start the test procedure
to check all is properly installed and function as expected. From a python
shell, you can type::

    >>> import dana
    >>> dana.test()
    ...........................................................................
    ..........................................................................
    ----------------------------------------------------------------------
    Ran 149 tests in 2.950s

If an error is detected, you should save the output and file a bug report at:
https://gforge.inria.fr/tracker/?group_id=628 giving your python, numpy and
scipy versions::

    >>> import sys, numpy, scipy
    >>> print sys.version
    2.7.1 (r271:86882M, Nov 30 2010, 10:35:34) 
    [GCC 4.2.1 (Apple Inc. build 5664)]
    >>> print numpy.__version__
    1.5.1
    >>> print scipy.__version__
    0.9.0b1
    >>> print dana.__version__
    0.3.3
    
Hopefully, some maintainer will take care of the problem and contact you if
necessary.
