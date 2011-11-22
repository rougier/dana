#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
import unittest
import numpy as np
from dana import Group
from tools import np_equal
from dana import ConnectionError
from dana import SharedConnection, SparseConnection, DenseConnection


class GroupDefault(unittest.TestCase):
    def setUp(self):
        self.Z = Group()
    def test_shape(self):
        assert self.Z.shape == ()
    def test_dtype(self):
        assert self.Z.dtype == np.dtype([('f0',float),])
    def test_size(self):
        assert self.Z.size == 1
    def test_fieldname(self):
        assert self.Z.f0 == self.Z.f0

class GroupShape(unittest.TestCase):
    def setUp(self):
        self.Z = Group((3,5))
    def test_shape(self):
        assert self.Z.shape == (3,5)
    def test_size(self):
        assert self.Z.size == 15
    def test_len(self):
        assert len(self.Z) == 3


class GroupFill(unittest.TestCase):
    def setUp(self):
        self.Z = Group((3,5), fill=1.2)
    def test_fill(self):
        assert 1-(self.Z.f0-np.ones((3,5))*1.2).all()

class GroupMask(unittest.TestCase):
    def setUp(self):
        self.Z = Group((3,3))

    def test_mask_true(self):
        self.Z.mask = True
        self.Z[...] = 1
        self.Z.setup()
        assert np_equal(self.Z.f0,np.ones((3,3)))

    def test_mask_1(self):
        self.Z.mask = 1
        self.Z[...] = 1
        self.Z.setup()
        assert np_equal(self.Z.f0,np.ones((3,3)))

    def test_mask_false(self):
        self.Z.mask = False
        self.Z[...] = 1
        self.Z.setup()
        assert np_equal(self.Z.f0,np.zeros((3,3)))

    def test_mask_0(self):
        self.Z.mask = 0
        self.Z[...] = 1
        self.Z.setup()
        assert np_equal(self.Z.f0,np.zeros((3,3)))

    def test_mask_shape_0(self):
        self.Z.mask = np.zeros((3,3))
        self.Z[...] = 1
        self.Z.setup()
        assert np_equal(self.Z.f0,np.zeros((3,3)))

    def test_mask_shape_1(self):
        self.Z.mask = np.ones((3,3))
        self.Z[...] = 1
        self.Z.setup()
        assert np_equal(self.Z.f0,np.ones((3,3)))

    def test_mask_shape_2(self):
        mask = np.ones((3,3))
        mask[1,1] = 0
        self.Z.mask = mask
        self.Z[...] = 1
        self.Z.setup()
        assert np_equal(self.Z.f0,mask)


class GroupDtype(unittest.TestCase):
    def setUp(self):
        self.Z = Group((5,5), dtype=[('x',float), ('y',float)])
    def test_dtype(self):
        assert self.Z.dtype == np.dtype([('x',float), ('y',float)])
    def test_contiguity(self):
        assert self.Z.x.flags['C_CONTIGUOUS']
        assert self.Z.y.flags['C_CONTIGUOUS']

class GroupAddItem(unittest.TestCase):
    def setUp(self):
        self.Z = Group((5,5), dtype=[('x',float), ('y',float)])
    def test_int(self):
        self.Z['z'] = 1
        assert self.Z.z.shape == (5,5)
        assert self.Z.z.dtype == np.int
    def test_float(self):
        self.Z['z'] = 1.0
        assert self.Z.z.shape == (5,5)
        assert self.Z.z.dtype == np.float

class GroupDelItem(unittest.TestCase):
    def setUp(self):
        self.Z = Group((5,5), dtype=[('x',float), ('y',float)])
    def test_del(self):
        del self.Z['y']
        assert self.Z.dtype == np.dtype([('x',float),])

class GroupSetItem(unittest.TestCase):
    def setUp(self):
        self.Z = Group((5,5), dtype=[('U', float), ('V', int)])
    def test_setitem_1(self):
        self.Z[0,0] = 2
        assert self.Z[0,0] == (2,2)
    def test_setitem_2(self):
        self.Z[0,0] = 1,2
        assert self.Z[0,0] == (1,2)
    def test_setitem_3(self):
        self.Z[:,:] = 1,2
        assert self.Z.U.sum() == 1*self.Z.size
        assert self.Z.V.sum() == 2*self.Z.size
    def test_setitem_4(self):
        self.Z[...] = 1,2
        assert self.Z.U.sum() == 1*self.Z.size
        assert self.Z.V.sum() == 2*self.Z.size
    def test_Group_setitem_5(self):
        def _set_(): self.Z[0,0] = 1,2,3
        self.assertRaises(ValueError, _set_)

class GroupGetItem(unittest.TestCase):
    def setUp(self):
        self.Z = Group((5,5), dtype=[('U', np.double), ('V', np.int32)])
    def test_getitem_1(self):
        a,b = self.Z[0,0]
        assert type(a) is np.double
        assert type(b) is np.int32
    def test_getitem_2(self):
        assert self.Z[:2,:2].shape == (2,2)
        assert self.Z[:2,:2].dtype == np.dtype([('U', np.double), ('V', np.int32)])

class GroupReshape(unittest.TestCase):
     def setUp(self):
         self.Z = Group((5,5), dtype=[('x',float), ('y',float)])
     def test_reshape(self):
         Z = self.Z.reshape((25,))
         assert Z.x.base is self.Z.x
         assert Z.y.base is self.Z.y

class GroupSubGroup(unittest.TestCase):
     def setUp(self):
         self.Z = Group((5,5), dtype=[('x',float), ('y',float)])
     def test_subgroup_1(self):
         S = self.Z.subgroup('x')
         assert S.base is self.Z
     def test_subgroup_2(self):
         S = self.Z('x')
         assert S.base is self.Z
     def test_subgroup_connections(self):
         src = np.ones((3,3))
         dst = Group((3,3), '''U = A+B+V;  V = C+D+Z; Z = E
                               A;B;C;D;E''')
         SparseConnection(src,dst('A'),np.ones((1,1)))
         SparseConnection(src,dst('B'),np.ones((1,1)))
         SparseConnection(src,dst('C'),np.ones((1,1)))
         SparseConnection(src,dst('D'),np.ones((1,1)))
         SparseConnection(src,dst('E'),np.ones((1,1)))
         Z = dst('U')
         links = []
         for c in Z.connections:
             links.append(c.target_name)
         assert 'A' in links
         assert 'B' in links
         assert 'C' in links
         assert 'D' in links
         assert 'E' in links


class GroupFunctions(unittest.TestCase):
    def test_Group_asarray(self):
        G = Group((5,5), dtype=[('U', float), ('V', int)], fill=1)
        A = G.asarray()
        assert A.shape == (5,5)
        assert A.dtype == np.dtype([('U', float), ('V', int)])
        assert A['U'].sum() == 25
        assert A['V'].sum() == 25


class GroupConnections(unittest.TestCase):
     def test_1(self):
         G = Group(5, 'V = I; I')
         Z = np.ones(5)
         DenseConnection(Z, G('I'), 1)
         DenseConnection(Z, G('I'), 2)
         G.run()
         assert np_equal(3*np.ones(5),G('V'))
         

if __name__ == "__main__":
    unittest.main()
    
