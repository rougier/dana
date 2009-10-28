
BCM Learning Rule
-------------------------------------------------------------------------------

.. contents::
   :depth: 1
   :local:


BCM is a distributed learning rule, in the sense that one unit can learn
something from its input by itself. A BCM unit tend to become selective to one
precise stimulus of the input, that is, having a strong activity for one
stimulus and a weak one for the rest. In contrast, the SOM learning algorithm
needs to define a winner inside a population of units to decide of the learning
rate of each unit.

The state of a BCM unit is described by two numerical values :

    1. *c*, the instantaneous activity of the unit
    2. *theta*, the long-term potentiation/long-term depression (LTP/LTD) threshold.

The basis of the BCM learning rule is to consider that the activity *c* relative
to the threshold *theta* defines the sign of the weight change. If the activity
is above the threshold, the learning operates as a potentiation and strengthens
the weights. Otherwise, the learning operates as a depression and weakens the
weights.


Here is a simple example of BCM. A set of *N* units are learning to be
selective to stimuli from a learning set. A *source* group is displays the
stimuli, a *bcm* group represent the learning units, and weighted links are
propagating activity from *source* to *bcm*.

.. code-block:: python

    import numpy, dana, time, dana.pylab
    from random import choice
    
    N = 10
    source   = dana.group((N,1), name='source')
    bcm      = dana.group((N,1), fields=['c','t'], name='bcm')
    bcm['c'] = 1
    bcm['t'] = 1
    
    K = numpy.random.random(bcm.shape + source.shape)
    bcm.connect(source['V'],'F',K,shared=False)
    

The set of stimuli is composed of *N* orthogonal stimuli. They are all
normalized.

.. code-block:: python

    stims = numpy.identity(N)

Three parameters are expressing variation speed of several variables of the
model :

    1. *TAU* : variation speed of the instantaneous activity
    2. *TAU_BAR* : variation speed of the *theta* threshold
    3. *ETA* : learning rate

In this example, *TAU = 1* which means that the activity is the instantaneous
integration of the feedforward activity. However, we could use a smaller value
for *TAU*, but we would have to present each stimulus for more than one step so
the unit have enough time to integrate the activity.

.. code-block:: python

    TAU     = 1.0
    TAU_BAR = TAU * 0.1
    ETA     = TAU_BAR * 0.1

    bcm.constant['tau']     = TAU
    bcm.constant['tau_bar'] = TAU_BAR
    bcm.constant['eta']     = ETA

We define the equations for the variations of the activity, the threshold and
the weights.

.. code-block:: python

    bcm.equation['c']       = "c + (F - c) * tau"
    bcm.equation['t']       = "t + (c**2 - t) * tau_bar"
    bcm.equation['F']       = "W + pre['V'] * post['c'] * (post['c'] - post['t']) * eta"

Finally, we run a simulation by presenting 1000 stimuli uniformly chosen from
the training set, and display the result.

.. code-block:: python

    n = 10000
    t = time.clock()
    for i in range(n):
        source['V'] = choice(stims).reshape(source.shape)
        bcm.compute()
        bcm.learn()
    print time.clock()-t


    view = dana.pylab.view([source['V'], bcm['c']])
    view.show()
    
