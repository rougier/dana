========
Glossary
========

.. glossary::

**layer**
    A group with a single field.

----

**group**
    A group is equivalent to a numpy structured array with a different memory
    layout. Array layout is such that the data pointer points to one block of
    *n* items, where each item is described by its dtype. In a group, each
    element of the dtype is a single block of *n* items.

