.. _image:

================
Image processing
================

A lot of standard image processing technics are based upon homogeneous
convolution of individual pixels with surrounding regions (`Gaussian blur
<http://en.wikipedia.org/wiki/Gaussian_blur>`_, `Sobel operator
<http://en.wikipedia.org/wiki/Sobel_operator>`_, `Cross operator
<http://en.wikipedia.org/wiki/Roberts_Cross>`_, etc.). DANA is perfectly suited
for such technics offering easy manipulation of kernel functions.


Loading an image
================

We'll use the `Imaging <http://www.pythonware.com/products/pil/>`_ library to
load the standard lena image into a numpy array. The shape of the array is
either *width x height* for gray-level images or *width x height x [3,4]*
for color images (RGB or RGBa).

.. code-block:: python

   import Image, dana, numpy
   
   image = numpy.asarray(Image.open('lena.png'))/256.0

In our case, the image is the standard Lena:

.. image:: _static/lena-RGB-small.png

We thus create a DANA group with 4 values to hold each of the red, green and
blue channels + a luminance value that needs to be computed based on available
channels:

.. code-block:: python

   I = dana.group(I.shape[:-1], keys=['L','R','G','B'])
   I.R = image[..., 0]
   I.G = image[..., 1]
   I.B = image[..., 2]
   I.L = 0.212671*I.R + 0.715160*I.G + 0.072169*I.B


Sobel operator
==============

The Sobel operator is used for edge detection and is based upon a
differentiation operator that approximate the gradient of the image
intensity. Technically, this is made using two convolutions (one horizontal
and one vertical) with a minimal size kernel (3x3) to get local gradients from
which we can compute the gradient magnitude:

.. code-block:: python

   # Create a group for (S)obel filter
   S = dana.zeros(I.shape)

   # Connect filter group to luminance
   S.connect(I.L, numpy.array([[-1, 0,+1],
                               [-2, 0,+2],
                               [-1, 0, 1]]), 'Gx', shared=True)
   S.connect(I.L, numpy.array([[+1,+2,+1],
                               [ 0, 0, 0],
                               [-1,-2,-1]]), 'Gy', shared=True)

   # Compute Sobel filters
   S.dV = '-V + sqrt(Gx*Gx+Gy*Gy)'
   S.compute()


Gaussian blur
=============

Gaussian blur is based upon the convolution of the image using a Gaussian
kernel. In the example below, we use a 5x5 Gaussian kernel and we apply it on
the three channels.

.. code-block:: python

   # Create a group for gaussian (B)lur
   B = dana.zeros(I.shape[:-1], keys=['R','G','B'])

   # Connect filter group to blur
   K = np.array([[1, 4, 7, 4,1],
                 [4,16,26,16,4],
                 [7,26,41,16,7],
                 [4,16,26,16,4],
                 [1, 4, 7, 4,1]])/273.0
   B.connect(I.R, K, 'r', shared=True)
   B.connect(I.G, K, 'g', shared=True)
   B.connect(I.B, K, 'b', shared=True)

   # Compute Gaussian blur
   B.dR = '-R +r'
   B.dG = '-G +g'
   B.dB = '-B +b'
   B.compute()


Visualization
=============

Finally, we can see results using matplotlib. Note that to be able to visualize
B as a standard RGB array, it requires some operations.

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(15,5))
   plt.subplot(1,3,1)
   plt.title('Original image')
   plt.imshow(image, origin='upper', interpolation='bicubic')
   plt.subplot(1,3,2)
   plt.title('Sobel filter')
   plt.imshow(S.V, origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
   plt.subplot(1,3,3)
   plt.title('Gaussian blur')
   plt.imshow(B.asarray().view(float).reshape(B.shape+(3,)),
             origin='upper', interpolation='bicubic', cmap=plt.cm.gray)
   plt.show()


.. image:: _static/image-filtering.png
