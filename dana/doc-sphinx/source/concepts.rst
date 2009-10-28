.. title:: Concepts

Concepts                                                                       
===============================================================================

The computational  paradigm supporting  the DANA framework  is grounded  on the
notion of  a unit that is  a set of arbitrary  values that can  vary along time
under the influence of other units and learning. Each unit can be linked to any
other unit (including  itself) using a weighted  link and a group is  a one- or
two-dimensional set  of homogeneous units. The  DANA framework offers  a set of
core objects  needed to design and  run such models. However,  what is actually
computed by a unit and what is learned is the responsibility of the modeler who
is in charge of describing the  equation governing the behavior of units groups
over time.

.. list-table:: 
   :widths: 25 75
   :header-rows: 0

   * - .. image:: _static/groups.png

     - * Each unit group A has three values (A1,A2 and A3)
       * Each unit group B has two values (B1 and B2)
       * Connection originating from layer A1 (source) to layer B1 (target)
         represent 
    

dana is based on vectorized computation, this means that the building block of a
model it not a neuron alone but a group of similar neurons organized
topologically. Vectorized computations are carried out using the `numpy
<http:/www.numpy.org>`_ scientific package thay offers a powerful N-dimensional
array object as well as sophisticated (broadcasting) functions. Further numpy
information can be obtained from:

* `NumPy and SciPy documentation page <http://www.scipy.org/Installing_SciPy>`_
* `NumPy Tutorial <http://www.scipy.org/Tentative_NumPy_Tutorial>`_
* `NumPy functions by category <http://www.scipy.org/Numpy_Functions_by_Category>`_
* `NumPy Mailing List <http://www.scipy.org/Mailing_Lists>`_


NumPy introduction                                                             
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2

   group
   link
   equation
