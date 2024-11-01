import unittest

from halo import Grid3x3
from mpi4py import MPI
import numpy as np
import pytest

"""
    3x3 grid.
    | 0 | 1 | 2 |
    | 3 | 4 | 5 |
    | 6 | 7 | 8 |
"""

class Test3x3(unittest.TestCase):
    grid = Grid3x3()
    partition = grid.partition()

    @pytest.mark.mpi(min_size=9)
    def test_partitioning(self):
        assert self.partition[0] == MPI.COMM_WORLD.Get_rank()
        
    @pytest.mark.mpi(min_size=9)
    def test_relative_at(self):
        assert self.grid.relative_at(4, "W") == 3
        assert self.grid.relative_at(4, "S") == 7
        assert self.grid.relative_at(4, "E") == 5
        assert self.grid.relative_at(4, "N") == 1

        assert self.grid.relative_at(0, "W") == 2
        assert self.grid.relative_at(0, "S") == 3
        assert self.grid.relative_at(0, "E") == 1
        assert self.grid.relative_at(0, "N") == 6

    @pytest.mark.mpi(min_size=9)
    def test_halo(self):
        KGO = np.array([3, 7, 5, 1])

        owned_indices = np.array([4])
        halo = self.grid.build_halo(owned_indices)

        np.testing.assert_allclose(halo, KGO, err_msg="Halo KGO fail: \n%s\n----------------------------\n%s")


    @pytest.mark.mpi(min_size=9)
    def test_3x3_9PE(self):
        """
        | 0 | 1 | 2 |
        | 3 | 4 | 5 |
        | 6 | 7 | 8 |

        With mpi_size = 9, each cell is an MPI region.
        Example for rank = 1:

        source indices => [1]
        target indices => [ 2, 3, 1, 6 ]   (W, S, E, N halo region).
        """
        # ---
        #np.testing.assert_allclose([], KGO_VERTS, err_msg="Connection KGO fail: \n%s\n----------------------------\n%s" % (verts, KGO_VERTS))

if __name__ == '__main__':
    unittest.main()




