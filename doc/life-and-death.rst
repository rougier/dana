.. currentmodule:: dana

===============================================================================
Life and Death                                                                 
===============================================================================

.. only:: html

   .. contents::
      :local:
      :depth: 2


Dead units                                                                     
===============================================================================

Any group can be given a mask attribute that indicates which units are functional
and which units are non-functional::

   >>> G = dana.zero(5,'V')
   >>> G.mask = np.ones(5,dtype=bool)
   >>> G.mask[2] = False
   >>> G.V = 1
   >>> print G.V
   [1. 1. -- 1. 1.]




Dead connections                                                               
===============================================================================

