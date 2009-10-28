=====
Group
=====

A group is a two dimensional array  of homogenous elements that can have one to
several values of any numpy type:

.. _numpy-type-table:

========== ====================================================================
Data type  Description
========== ====================================================================
bool 	   Boolean (True or False) stored as a byte
int 	   Platform integer (normally either int32 or int64)
int8 	   Byte (-128 to 127)
int16 	   Integer (-32768 to 32767)
int32      Integer (-2147483648 to 2147483647)
int64      Integer (9223372036854775808 to 9223372036854775807)
uint8 	   Unsigned integer (0 to 255)
uint16     Unsigned integer (0 to 65535)
uint32 	   Unsigned integer (0 to 4294967295)
uint64 	   Unsigned integer (0 to 18446744073709551615)
float 	   Shorthand for float64.
float32    Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
float64    Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
complex    Shorthand for complex128.
complex64  Complex number, represented by two 32-bit floats (real and imaginary
           components)
complex128 Complex number, represented by two 64-bit floats (real and imaginary
           components)
========== ====================================================================

Each value is organized into a  compact numpy array of group dimensions and any
value can be accessed using the relevant key.

.. toctree::
   :maxdepth: 1

   group-creation
   group-manipulation
