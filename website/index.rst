.. include:: header-about.txt
.. include:: footer.txt


What is DANA ?
==============

Dana is a python framework for distributed, asynchronous, distributed and
adaptive computing. The computational paradigm supporting the dana framework is
grounded on the notion of a unit that is a essentially a set of arbitrary
values that can vary along time under the influence of other units and
learning. Each unit can be connected to any other unit (including itself) using
a weighted link and a group is a structured set of such homogeneous units.

.. image:: group.png
   :alt:   Dana basic concepts

More formally, we can write the following definitions:

* A unit is a set of one to several values (Vᵢ, i ∈ ℕ), each of them being
  potentially described by an equation.
* A group is a structured set of one to several homogeneous units.
* A layer is a subset of a group restricted to a unique value Vᵢ.
* A layer is a group.
* A connection links a source layer to a target layer and may have an equation
  describing its evolution along time according to source and target.
* A group can be connected to any other group value including itself.

The dana framework offers a set of core objects needed to design and run such
models. However, what is actually computed by a unit and what is learned is the
responsibility of the modeler who is in charge of describing the equation
governing the behavior of units groups over time and/or learning.
