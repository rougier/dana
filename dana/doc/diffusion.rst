
Diffusion process
-------------------------------------------------------------------------------
.. contents::
   :depth: 1
   :local:

Theory
++++++

**Text from wikipedia entry** [#]_

.. figure:: _static/diffusion.png
   :figclass: figure-right

   50x50 cells after 2 seconds

The heat equation is an important partial differential equation which describes
the distribution of  heat (or variation in temperature) in  a given region over
time. For a function u(x,y,t) of two spatial variables (x,y) and the time
variable t, the heat equation is

.. math::

    \frac{\partial u}{\partial t} -k\left( \
     \frac{\partial^2u}{\partial x^2} \
    +\frac{\partial^2u}{\partial y^2}\right) = 0

or equivalently

.. math::

    \frac{\partial u}{\partial t} = k \nabla^2 u

where k is a constant.

The heat equation is of fundamental importance in diverse scientific fields. In
mathematics, it is the prototypical parabolic partial differential equation. In
statistics, the  heat equation is connected  with the study  of Brownian motion
via the Fokker-Planck equation. The  diffusion equation, a more general version
of the heat equation, arises in connection with the study of chemical diffusion
and other related processes.


Implementation
++++++++++++++

Using  the same setup  as for  the game  of life,  we can  also modify  the *V*
equation to implement  a diffusion process where each  cell slowly acquires the
mean value of its 8 neighbours.

.. code-block:: python

   arena.constant = {'k': 1}
   arena.equation['V'] = 'V+dt*k*(N/8-V)'

To get some interesting effect, we maintain the four borders of the arena at
different values and iterate the diffusion process.

.. code-block:: python

   for i in range(200):
       arena.compute(0.01)
       arena['V'][0,:] = arena['V'][n-1,:] = arena['V'][:,0] = 1
       arena['V'][:,n-1] = -1


And we display the result using pylab.

.. code-block:: python

   view = dana.pylab.view(arena['V'], vmin=-1, vmax=1)
   pylab.xticks('')
   pylab.yticks('')
   view.show()


----------

*Notes*

    .. [#] http://en.wikipedia.org/wiki/Heat_equation
