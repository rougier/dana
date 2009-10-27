=====
Group
=====

A group represents a matrix of homogeneous elements that can have an arbitrary
number of values whose name and type must be described at group creation. These
values can be later accessed using either dictionary lookups such as
``group['x']`` and ``group['y']`` or accessed as members using ``group.x`` and
``group.y``. A group can also be connected to another group (source) using a
connection kernel. These connections can be later used when computing new values
of the group using dedicated equation.


-----

.. _group.group:

**__init__** ``(self, shape=(1,1), dtype=numpy.float32, keys=['V'],  mask=True, name='', fill=None)``

  Create a new group.

  * ``shape`` : Shape of group as a tuple of two ints 
  * ``dtype`` : Desired data type
  * ``keys``  : Name of the different values
  * ``mask``  : Group mask indicated active and inactive elemnts
  * ``name``  : Group name
  * ``obj``   : Array like object to create group from
  * ``fill``  : Fill value


-----

.. _group.connect:

**connect** ``(self, source, name, kernel, mask=None, shared=True)``

  Connect group to a source group using specified kernel and mask.

  * ``source`` : Source group or array
  * ``name``   : Name of the link
  * ``kernel`` : Kernel array to be used for linking source to self
  * ``shared`` : Whether the kernel is shared among group elements


-----

.. _group.disconnect:

**disconnect** ``(self, name)``

  Disconnect an existing link.

  * ``name`` : Name of an existing link


-----

.. _group.compute:

**compute** ``(self, dt=0.1)``

  Update group values according to value equations and dt

  * ``dt`` : Elementary time step


-----

.. _group.learn:

**learn** ``(self, dt=0.1)``

  Update group link values according to link equations and dt

  * ``dt`` : Elementary time step
