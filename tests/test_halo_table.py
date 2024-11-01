import unittest

from meshing import Mesh
from halo import HaloTable
import logger

from mpi4py import MPI
import numpy as np
import pytest


class TestHaloTable(unittest.TestCase):
    def test_halo_table(self):
        KGO = np.array([1])

        # Which PE owns what.
        partition_mapping = np.array([0, 1])
        # Halo indices.
        halo_pe0 = np.array([1])

        table_pe0 = HaloTable(partition_mapping, halo_pe0)

        np.testing.assert_allclose(table_pe0.table, KGO, err_msg="HaloTable KGO fail: \n%s\n----------------------------\n%s")

    def test_simple_halo(self):
        # Simplest case of a mesh:
        # | 0 | 1 |
        # Split into two PEs. 0 and 1, with a halo depth of 1

        pe0 = Mesh(np.array([0]), np.array([1]))
        pe1 = Mesh(np.array([1]), np.array([0])) 

        field0 = pe0.make_field()
        # Data is [own, halo]
        field0.data[0] = 100
        field0.data[1] = -100

        field1 = pe1.make_field()
        field1.data[0] = 200
        field1.data[1] = -200



        


