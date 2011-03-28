Copyright (C) 2011 Nicolas P. Rougier.

Permission is granted to copy, distribute and/or modify this document under the
terms of the  GNU Free Documentation License, Version 1.3  or any later version
published  by the  Free Software  Foundation;  with no  Invariant Sections,  no
Front-Cover Texts, and no Back-Cover Texts.

A  copy  of  the  license  is  included  in  the  section  entitled  "GNU  Free
Documentation License".

===============================================================================
Preface                                                                        
===============================================================================
.. contents::
   :local:



What is in this handbook ?                                                     
===============================================================================
This book includes the following chapters:

* Chapter :doc:`intro` gives a general overview of the DANA computing
  framework
* Chapter :doc:`install` explains how to install DANA on your machine.
* Chapter :doc:`quickstart` gives a rapid overview of main concepts related to
  DANA.
* Chapter :doc:`numpy-to-dana` explains conceptual differences between array and group.
* Chapter :doc:`model` explains what are model and equations and how to use them.
* Chapter :doc:`connection` gives details on group connections
* Chapter :doc:`learning` gives main concepts related to learning
* Chapter :doc:`life-and-death` explains what are dead units and dead connections
* Chapter :doc:`time` details time management and timers
* Chapter :doc:`advanced` introduces advanced dana concepts
* Chapter :doc:`examples` comments on examples from various scientific domain.
* Chapter :doc:`api` Application Programming Interface
* Chapter :doc:`faq` gives answers to frequently asked questions
* Chapter :doc:`glossary` explains terms used in this book
* Chapter :doc:`license` explains what you can and cannot do with this book.



Who should read this book ?                                                   
===============================================================================

You  should read this  book if  you intent  to develop  models using  the DANA
computing framework and  especially if your models belong  to the computational
neuroscience domain. DANA  is a python library and  depends heavily on external
library  such as  numpy  and scipy.  However,  this book  does  not provide  a
tutorial  to the  python language  neither an  introduction to  the numpy/scipy
libraries. If you're  unfamiliar with both of them,  you'encouraged to document
yourself first  by considering  external resources for  both python,  numpy and
scipy.

If you're unfamiliar  with python, have a look first at  the very nice tutorial
by Mark  Pilgrim (which is also available  as a book). Numpy  user guide should
gives  you the  main concepts  related  to vectorized  computation while  scipy
tutorial may be considered optional but worth reading anyway.

**Python**

* Python website : http://www.python.org
* Python tutorial: http://diveintopython.org

**Numpy**

* Numpy website: http://numpy.scipy.org
* Numpy tutorial: http://docs.scipy.org/doc/numpy/user/
* Numpy manual: http://docs.scipy.org/doc/numpy/reference/

**Scipy**

* SciPy website: http://www.scipy.org
* SciPy tutorial: http://docs.scipy.org/doc/scipy/reference/tutorial/index.html
* SciPy manual: http://docs.scipy.org/doc/scipy/reference


Conventions used in this book                                                  
===============================================================================

A lot of examples are given throughout the book and the may be related to
either a regular shell, a python shell or an ipython shell. You can easily
distinguish them by the prompt they use:

**System shell**::

    $ 

**Python/IPython shell**::

    >>> 

Furthemore, since numpy, scipy and matplotlib libraries are extensively used
throughout the whole book, they will respectively referred as ``np``, ``sp``,
and ``plt`` and are supposed to have been imported as::

    >>> import numpy as np
    >>> import scipy as sp
    >>> import matplotlib.pyplot as plt



About this book                                                                
===============================================================================

This book has been  written using `Sphinx  <http://sphinx.pocoo.org/>`_ and was
last generated on |today|.

