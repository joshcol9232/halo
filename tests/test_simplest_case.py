import unittest

import halo
import numpy as np

# Square basis - simplest case
class TestSimplestCase(unittest.TestCase):
    def test_square_k1(self):
        KGO_VERTS = np.array([[0., 0.],
                              [1., 0.],
                              [0., 1.],
                              [1., 1.]])
        # ---
        np.testing.assert_allclose([], KGO_VERTS, err_msg="Connection KGO fail: \n%s\n----------------------------\n%s" % (verts, KGO_VERTS))

if __name__ == '__main__':
    unittest.main()




