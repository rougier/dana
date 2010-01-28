===========
Shared Link
===========

When a connection kernel has been specified as a two-dimensional array, it is
possible to specify whether the link is shared or not among target elements. A
shared link means that the kernel won't be expanded internally into a four
dimensional array. This usually saves a lot of memory, speeds-up computations in
most cases but prevent any learning to occur on this link.

Here is some benchmarks showing the computation time of a **weighted sum** using
a kernel of size 25x25 considering a source size of 50x50 elements:

=========== =========== =========== =========== ============= ================
Source size Target size Kernel size Kernel type Shared status Computation Time
=========== =========== =========== =========== ============= ================
50x50        50x50      25x25       Constant    True          0.5 seconds
50x50        50x50      25x25       Random      True          8.6 seconds
50x50        50x50      25x25       Constant    False         10.6 seconds
50x50        50x50      25x25       Random      False         10.6 seconds
=========== =========== =========== =========== ============= ================

The gain in computation time when links are shared comes from the internal use
of a convolution product between source and kernel. Furthemore, the kernel is
decomposed using a singular value decomposition and when the kernel is symetric,
the convolution is also decomposed as a product of two one-dimensional kernels.
When source and and target are of different shapes, source is zoomed in or out
as to match target size and convolution is computed.

