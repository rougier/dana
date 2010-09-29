.. _dnf:

=====================
Dynamic Neural Fields
=====================

Dynamic neural fields are neural models that describe the spatio-temporal
evolution of a population of neurons seen as a spatial continuum. These models
have been extensively studied in [Wilson:1972]_ , [Wilson:1973]_, [Amari:1977]_
and [Taylor:1999]_ extended these studies in the two-dimensional case. We will
use notations introduced by Amari where a neural position is labelled by a
vector x.

.. math::
   :label: eq-1

   \frac{1}{\alpha} \frac{\partial U(x,t)}{\partial t} = -u(x,t) +
   \int_{-\infty}^{+\infty}W(|x-y|)~f[U(y)]dy + h + I(x,t)

with
  * U(x,t) is the potential of a neural population at position x and time t.
  * W(x) is a neighborhood function
  * f(x) is the firing rate of a single neuron.
  * α is the temporal decay of the synapse.
  * I(x,t) is the input at position x

Usually, the neighborhood function W is a difference of Gaussian
(a.k.a. Mexican hat) with short range excitations and long range-inhibitions.


Discretization
==============

We begin by setting simulation parameters

.. code-block:: python

   import dana, numpy

   N = 60
   dt = 0.1
   alpha = 10.0
   tau = 1.0
   h = 0.0

and we create a group for the input (I) and another one for the output (U).

.. code-block:: python

   I = dana.zeros((n,n))
   U = dana.zeros((n,n))


Connections
===========

We first connect the input to the output using a sparse one-to-one connection.

.. code-block:: python

   U.connect(I.V, numpy.ones((1,1)), 'I', sparse=True)

Then we connect output to itself using a (shared) difference of Gaussian.

.. code-block:: python

   K = 1.25*dana.gaussian((2*n+1,2*n+1),0.1) \
     - 0.7*dana.gaussian((2*n+1,2*n+1),1)
   U.connect (U.V, K, 'L', shared=True)


Output equation
===============

.. code-block:: python

   U.dV = '-V + maximum(V+dt/tau*(-V+(L/(N*N)*10*10+I+h)/alpha),0)'


Simulation
==========

.. code-block:: python

   I.V = dana.gaussian((N,N), 0.2, ( 0.5, 0.5))
   I.V += dana.gaussian((N,N), 0.2, (-0.5,-0.5))
   I.V += (2*numpy.random.random((N,N))-1)*.05

   for i in range(250):
       focus.compute(dt)


Visualization
=============

.. code-block:: python

   dana.pylab.view([input.V, focus.V]).show()


.. figure:: _static/cnft.png

   Continum Neural Field Theory



References
==========

.. [Amari:1977] S.-I. Amari, *Dynamic of pattern formation in
                lateral-inhibition type neural fields*,
                Biological Cybernetics, 27:77-88, 1977.

.. [Wilson:1972] H.R. Wilson and J.D. Cowan *Excitatory and inhibitory
                 interactions in localized populations of model neurons*,
                 Biophysical Journal, 12:1-24, 1972.

.. [Wilson:1973] H.R. Wilson and J.D. Cowan. *A mathematical theory of the
                 functional dynamics of cortical and thalamic nervous tissue.*
                 Kybernetik, 13:55–80, 1973.

.. [Taylor:1999] J.G. Taylor *Neural bubble dynamics in two dimensions:
                 foundations*, Biological Cybernetics, 80:5167-5174, 1999.
