.. title:: DANA Documentation




Documentation                                                                  
===============================================================================

Computational neuroscience  is a  vast domain of  research going down  from the
very  precise modeling  of a  single spiking  neuron, taking  into  account ion
channels and/or  dendrites spatial  geometry up to  the modeling of  very large
assemblies  of simplified  neurons that  are able  to give  account  of complex
cognitive functions.  DANA attempts to address this latter modeling activity by
offering a python  computing framework for the design  of very large assemblies
of neurons  using numerical and  distributed computations. However,  there does
not exist something as a unified model of neuron: if the formal neuron has been
established  some sixty years  ago, there  exists today  a myriad  of different
neuron models  that can be used within  an architecture. Some of  them are very
close  to the  original  definition while  some  others tend  to  refine it  by
providing extra  parameters or  variables to  the model in  order to  take into
account the great variability of  biological neurons. DANA makes the assumption
that a neuron is essentially a set  of numerical values that can vary over time
due to  the influence of other neurons  and learning. DANA aims  at providing a
constrained anfd consistent python  framework that guarantee this definition to
be enforced anywhere  in the model, i.e., no symbol,  no homonculus, no central
executive.


Programming guide                                                              
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 1

   installation
   quick-tour
   concepts
   examples


API Reference                                                                  
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 3

   api
