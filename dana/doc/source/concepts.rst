.. -*- coding: utf-8 -*-
.. include:: common.rst
.. title:: Concepts

Concepts                                                                       
===============================================================================

The computational  paradigm supporting  the DANA framework  is grounded  on the
notion of  a unit that is  a set of arbitrary  values that can  vary along time
under the influence of other units and learning. Each unit can be linked to any
other unit (including itself) using a weighted link and a group is a structured
set of homogeneous units.

.. list-table:: 
   :widths: 25 75
   :header-rows: 0

   * - .. image:: _static/group.png

     - * A :blue:`unit` is a set of one to several values (Vᵢ, i ∈ ℕ).
       * A :blue:`group` is a structured set of one to several homogeneous
         units.
       * A :blue:`layer` is a subset of a group restricted to a unique value
         Vᵢ.
       * A layer is a group.
       * A :blue:`link` is a weighted connection between a source unit towards
         a target unit.
       * Target units own their links.
       * A unit can be linked to any other unit including itself.

The DANA framework offers  a set of core objects needed to  design and run such
models. However, what is actually computed by a unit and what is learned is the
responsibility  of the  modeler who  is in  charge of  describing  the equation
governing the behavior of units groups over time.


.. .. list-table:: 
..    :widths: 50 50
..    :header-rows: 0

..    * - * Considering a group A with three layers  (A₁, A₂ & A₃), a group B with
..          two layers (B₁ & B₂) and a group  C with two layers (C₁ & C₂). Group B
..          is  connected  to layer  A₂  from  group A  and  layer  C₁ from  group
..          C. Updating of group B can be made based only on those links.

..      - .. image:: _static/link.png


Furthermore,  DANA is  based on  vectorized  computation, this  means that  the
building block of  a model it not a  unit alone but a group  of similar neurons
organized  topologically. Vectorized  computations  are carried  out using  the
`numpy`_ scientific  package thay offers a powerful  N-dimensional array object
as well  as sophisticated  (broadcasting) functions. Further  numpy information
can be obtained from:

* `NumPy and SciPy documentation page <http://www.scipy.org/Installing_SciPy>`_
* `NumPy Tutorial <http://www.scipy.org/Tentative_NumPy_Tutorial>`_
* `NumPy functions by category <http://www.scipy.org/Numpy_Functions_by_Category>`_
* `NumPy Mailing List <http://www.scipy.org/Mailing_Lists>`_


DANA Objects                                                                   
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2

   group
   link
   equation
