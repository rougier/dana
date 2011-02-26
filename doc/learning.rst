.. currentmodule:: dana

===============================================================================
Learning                                                                       
===============================================================================
.. contents::
   :local:
   :depth: 2


Since dana stands for Distributed Asynchronous Numerical & Adaptive
computation, it is time for us to discover how to make a model adaptive (and
yes, we also need to talk about this asynchronous thing...).

A model can be made adaptive by specifying a differential equation for any
connection that tell dana how to update connection weights. At this stage, it
is quite important to distinguish between the connection output and the
connection weights. A connection output is referred within a group equation by
the name of the group field that receives the connection output while the
actual weight matrix within a connection differential equation is referred with
the ``W`` letter and the equation is thus an equation of the form ``dW/dt = A``
where A is a valid python expression.



Basic concepts                                                                 
===============================================================================
Let us consider the simple example below::

    >>> G = ones(1, '''V1 = I1; I1
                       V2 = I2; I2''')
    >>> C1 = DenseConnection(G('V1'), G('I1'), np.ones(1) )
    >>> C2 = DenseConnection(G('V2'), G('I2'), np.ones(1), 'dW/dt = 1')

G is a group with four fields (``V₁``, ``V₂``, ``I₁`` and ``I₂``) and both ``I₁``
and ``I₂`` receives the output respectively from ``C₁`` and ``C₂`` connections.

Weights from the ``C₂`` connection possess an equation and are consequently
updated at each time step with a constant increase of ``1`` (from their
definition). ``C₁`` connection does not have an equation and consequently,
weights will remain constant during a simulation. Now, let's run the group for
a few iterations with ``dt=1``::

    >>> run(n=3)
    >>> print G.V1, G.V2
    [ 1.] [ 6.]
    >>> print C1.weights, C2.weights
    [[ 1.]] [[ 4.]]

We can observe (as expected) that both final ``V₁`` and ``V₂`` values are
different as well as weights from ``C₁`` and ``C₂`` connection. If we run
manually the simulation, we can check those are the expected values::

    t=0: V₁(0) = 1
         V₂(0) = 1
         W₁(0) = 1
         W₂(0) = 1

    t=1: V₁(1) = W₁(0)*V₁(0) = 1
         V₂(1) = W₂(0)*V₂(0) = 1
         W₁(1) = 1 
         W₂(1) = W₂(0)+dt*1 = 2

    t=2: V₁(2) = W₁(1)*V₁(1) = 1
         V₂(2) = W₂(1)*V₂(1) = 2
         W₁(2) = 1 
         W₂(2) = W₂(1)+dt*1 = 3

    t=3: V₁(3) = W₁(2)*V₁(2) = 1
         V₂(3) = W₂(2)*V₂(2) = 6
         W₁(3) = 1 
         W₂(3) = W₂(2)+dt*1 = 4



Pre-synaptic and post-synaptic activities                                      
===============================================================================
As explained in the previous chapter, a connection is made between a source
group and a target group and the differential equation governing weights
activity over time may used activities from either source or target group. Now,
consider the following situation::


    >>> source = Group(10, 'V')
    >>> target = Group(10, 'V;I')
    >>> C = DenseConnection(source('V'), target('I'), np.ones(1),
                            'dW/dt = V')

Does the ``V`` value relates to the source or to the target group ? To
disambiguate this kind of situation, dana provides the ``pre`` and ``post``
keyword for the definition of the equation of a connection. The ``pre`` relates
to the source and the ``post`` relates to the target. We an now re-write the
equation withtou any ambiguities::

    >>> C = DenseConnection(source('V'), target('I'), np.ones(1),
                            'dW/dt = post.V')
