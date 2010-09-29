.. include:: header.incl
.. include:: footer.incl

===============================================================================
DANA                                                                           
===============================================================================
-------------------------------------------------------------------------------
Distributed (Asynchronous) Numerical & Adaptive computing framework            
-------------------------------------------------------------------------------

DANA is a python computing framework based on numpy and scipy libraries.

The computational  paradigm supporting  the DANA framework  is grounded  on the
notion of  a unit that is  a set of arbitrary  values that can  vary along time
under the influence of other units and learning. Each unit can be linked to any
other unit (including itself) using a weighted link and a group is a structured
set of such homogeneous units.  The dana framework offers a set of core objects
needed to design  and run such models. However, what is  actually computed by a
unit and what is learned is the  responsibility of the modeler who is in charge
of describing  the equation  governing the behavior  of units groups  over time
and/or learning.

.. figure:: group.png
   
   **Figure 1.** Dana basic concepts.  A unit  is a set of one to several values
   (Vᵢ, i  ∈ ℕ),  a group  is a structured  set of  one to  several homogeneous
   units. A layer  is a subset of a  group restricted to a unique  value Vᵢ.  A
   layer is  a group.  A link is  a weighted  connection between a  source unit
   towards a  target unit.  Target groups own  their links.   * A group  can be
   linked to any other group value including itself.

