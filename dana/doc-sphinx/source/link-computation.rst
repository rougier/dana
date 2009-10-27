================
Link computation
================

A link is used to compute some values using a source group , a connection kernel
and a function from ℝ² into ℝ. This kernel can be specified either as a two
dimensional array or a four-dimensional one:

* When the kernel is specified as a two-dimensional array, it implicitely means
  that every element of the target group shares the same kernel.

* When the kernel is specified as a four-dimensional array, it explicitely means
  that each element in the target group possesses its own specific sub-kernel.


Two-dimensional kernel
======================

To understand how the computation is performed in case of a two dimensional
kernel, let us consider:

* a source group **S** of size (s₁,s₂)
* a target group **T** of size (t₁,t₂)
* a connection kernel **K** of size (k₁,k₂)
* an arbitrary function **f**: ℝ×ℝ→ℝ

and let **I** (of size (i₁,i₂)) be the output of the computation. Indepentently
of **T** and **K**, the result of link computation is always a two-dimensional
array of size (s₁,s₂), i.e. i₁=s₁ and i₂=s₂.

Any element I[i,j] of **I** is computed as a function (**f**) of the kernel
**K** and a sub-part of the source group **S** as show on the figure below:

.. image:: _static/link.png

[i′,j′] coordinates are computed using the [i,j] **T** coordinates translated
into **S** coordinates as follows:

* i′ = (i/t₁)*s₁
* j′ = (j/t₂)*s₂

To compute I[i,j], the kernel **K** is centered on these [i',j'] coordinates and
the function **f** is applied between **K** and the corresponding sub-part of
**S**.  Note that if some part of the kernel when mapped onto **S** is outside
of it, it is not taken into account.

.. math::

   \forall i,j \in [0,s_1[ \times [0,s_2[,
       i' &= \frac{i*t_1}{s_1} - \frac{k_1}{2}\\
       j' &= \frac{j*t_2}{s_2} - \frac{k_2}{2}\\
   I[i,j] &= \sum_{k=0}^{k_1-1} \sum_{l=0}^{k_2-1}
              f\left( S[i'-k,j'-l], K[k,l] \right)\\
   I[i,j] &= \sum_{k=0}^{k_1-1} \sum_{l=0}^{k_2-1}
              f \left( S[\frac{i*t_1}{s_1} - \frac{k_1}{2}-k,
                       \frac{j*t_2}{s_2} - \frac{k_2}{2}-l],  K[k,l] \right)


Four-dimensional kernel
=======================

If the kernel has been specified as a four-dimensional array, and considering a
source group **S** of size (s₁,s₂) and a target group **T** of size (t₁,t₂), the
size of **K** must be (s₁,s₂,t₁,t₂). Computing **I** is then straightforward:

.. math::

   \forall i,j \in [0,s_1[ \times [0,s_2[,
   I[i,j] &= \sum_{k=0}^{s_1-1} \sum_{l=0}^{s_2-1}
              f\left( S[k,l], K[i,j,k,l] \right)\\

Like in the two-dimensional kernel case,  the size of **I** is also (s₁,s₂).
