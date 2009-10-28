
Dynamic Neural Field
===============================================================================

Dynamic neural fields are neural models that describe the spatio-temporal
evolution of a population of neurons seen as a spatial continuum. These models
have been extensively studied in [Wilson:1972]_ , [Wilson:1973]_, [Amari:1977]_
and [Taylor:1999]_ extended these studies in the two-dimensional case. We will
use notations introduced by Amari where a neural position is labelled by a
vector x.

.. math::

  \tau \frac{\partial u(x,t)}{\partial t} = -u(x,t) +
  \int_{-\infty}^{+\infty}w(|x-y|)f[u(y)]dy + h + s(x,t) \mbox{~~~~~(1)}


where *u(x,t)* is interpreted as a neural field representing the population
activity at position *x* and time *t*, *τ* is a time constant, *f* is the firing
rate function of a single neuron, h is the resting potential of neurons,
*W(|x-y|)* is the connection strenght between population at position *x* and
population at position *y* and *s(x,t)* is the external input received at
position *x*. *M* is the domain of the ne



.. code-block:: python

   import numpy, dana

   # Simulation parameters
   n       = 40
   dt      = 0.1
   alpha   = 10.0
   tau     = 1.0
   h       = 0.0

   # Build groups
   input = dana.group((n,n), name='input')
   focus = dana.group((n,n), name='focus')

   # Connections
   Wi = numpy.ones((1,1))
   focus.connect(input['V'], 'I', Wi, shared=True)

   Wl = 1.25*dana.gaussian(2*n+1, 0.1*n) - 0.7*dana.gaussian(2*n+1, 1*n)
   focus.connect (focus['V'], 'L', Wl, shared=True)

   # Set Dynamic Neural Field equation
   focus.constant = {'tau':tau, 'alpha':alpha, 'h':h}
   focus.equation['V'] = 'maximum(V+dt/tau*(-V+(L+I+h)/alpha),0)'


   # Set input
   input['V']  = dana.gaussian(n, 0.1*n, ( 0.25, 0.25))
   input['V'] += dana.gaussian(n, 0.1*n, (-0.25,-0.25))
   input['V'] += (2*numpy.random.random((n,n))-1)*.05



Dynamic neural field equation 1nft` reads.

.. figure:: _static/cnft.png

   Continum Neural Field Theory



Bibliography
-------------------------------------------------------------------------------

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
