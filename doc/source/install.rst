.. highlight:: none

============
Installation
============

Prerequesites
=============

Dana is a python package and depends on a number of python components that can
be easy installed (see instructions on respective homepages).

Mandatory components
--------------------

* **numpy**, http://numpy.scipy.org

  NumPy is the fundamental package for scientific computing in Python. It is a
  Python library that provides a multidimensional array object, various derived
  objects (such as masked arrays and matrices), and an assortment of routines
  for fast operations on arrays, including mathematical, logical, shape
  manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic
  linear algebra, basic statistical operations, random simulation and much
  more.

* **scipy**, http://www.scipy.org

  SciPy is a collection of mathematical algorithms and convenience functions
  built on the Numpy extension for Python. It adds significant power to the
  interactive Python session by exposing the user to high-level commands and
  classes for the manipulation and visualization of data. With SciPy, an
  interactive Python session becomes a data-processing and system-prototyping
  environment rivaling sytems such as Matlab, IDL, Octave, R-Lab, and SciLab.

Optional components
-------------------

* **matplotlib**, http://matplotlib.sourceforge.net

  matplotlib is a python 2D plotting library which produces publication quality
  figures in a variety of hardcopy formats and interactive environments across
  platforms. matplotlib can be used in python scripts, the python and ipython
  shell (ala matlab or mathematica), web application servers, and six graphical
  user interface toolkits.


Optional tools
--------------

* **IPython**, http://ipython.scipy.org/moin/FrontPage

  The goal of IPython is to create a comprehensive environment for interactive
  and exploratory computing. To support, this goal, IPython has two main
  components: An enhanced interactive Python shell.  An architecture for
  interactive parallel computing.  All of IPython is open source (released
  under the revised BSD license). You can see what projects are using IPython
  here, or check out the talks and presentations we have given about IPython.



Installation in-place
=====================

Dana can be ran in-place without any installation and thus you might want to
experiment first with dana before you install it on your development
machine. To do this, add either the extracted dana source archive directory or
the compressed runtime egg to your ``PYTHONPATH``.

On Windows you can specify this from a command line::

   set PYTHONPATH c:\path\to\dana\;%PYTHONPATH%


On Mac OS X, Linux or on Windows under cygwin using bash::

   set PYTHONPATH /path/to/dana/:$PYTHONPATH
   export PYTHONPATH

or, using tcsh (or a variant)::

   setenv PYTHONPATH /path/to/dana/:$PYTHONPATH

If you have downloaded a runtime egg instead of the source archive, you would
specify the filename of the egg in place of ``dana/``.


Installation using setup.py
===========================

Local Installation
------------------

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


Global Installation
-------------------

To make dana available to all users you can install it into your Python's
site-packages directory.

From a command prompt on Windows, change into the extracted dana source
archive directory and type::

   python setup.py install


On Mac OS X and Linux you will need to do the above as a priveleged user; for
example using sudo::

   sudo python setup.py install



Installation from eggs
======================

If you have *setuptools* installed, you can install or upgrade to the latest
version of dana using easy_install::

   easy_install -U dana

On Mac OS X and Linux you may need to run the above as a priveleged user; for
example::

   sudo easy_install -U dana


Testing installation
====================

Once you have chosen your installation, you can test dana by importing and
running the ``test`` function::

   >>> import dana
   >>> dana.test()
   ...............................................................................
   ----------------------------------------------------------------------
   Ran 100 tests in 0.673s

   OK

If you some find some errors during the test, please fill a bug-report and send
it to `Nicolas Rougier <Nicolas.Rougier@loria.fr>`_.
