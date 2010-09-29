.. _faq:
.. highlight:: none

==========================
Frequently asked questions
==========================

What is DANA ?
==============
Please have a look at " :doc:`intro` " section of the documentation.


Differential equation does not update my variable
=================================================

Let us consider the following code::

   >>> y = 1
   >>> eq = dana.DifferentialEquation('dy/dt = y')
   >>> eq.run(y=1, t=1.0, dt=0.01)
   >>> print y
   1
