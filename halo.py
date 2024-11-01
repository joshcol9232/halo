import numpy as np

import logger
from mpi4py import MPI


class Grid3x3:
    def __init__(self):
        self.indices = np.array([0, 1, 2,
                                 3, 4, 5,
                                 6, 7, 8])
        self.size = 3

    def relative_at(self, source: int, direction: str) -> int:
        # E.g at 4, offset direction = S -> 7
        dir_offset = {"W": 0, "S": 1, "E": 2, "N": 3}
        #                    W  S  E  N
        mapping = np.array([[2, 3, 1, 6], [0, 4, 2, 7], [1, 5, 0, 8],
                            [5, 6, 4, 0], [3, 7, 5, 1], [4, 8, 3, 2],
                            [8, 0, 7, 3], [6, 1, 8, 4], [7, 2, 6, 5]])

        return mapping[source, dir_offset[direction]]

    def build_halo(self, owned_indices: np.array) -> np.array:
        return np.array([ self.relative_at(ow, di) for di in ["W", "S", "E", "N"] for ow in owned_indices ])

    def partition(self, comm=MPI.COMM_WORLD):
        # Available parition schemes:
        # "Individual" : each cell is an MPI rank. world size = 9.
        # "Stripes" : Each horizontal strip.
        num_ranks = comm.Get_size()

        if num_ranks == 9:
            owned_indices = np.array([self.indices[comm.rank]])
            halo_indices = self.build_halo(owned_indices)
            return owned_indices, halo_indices
        #elif num_ranks == 3:
        #    return self.indices[comm.rank : comm.rank + 3]
        else:
            raise ValueError("Grid3x3.partition: ERROR: Expected either 3 or 9 mpi ranks.")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    grid = Grid3x3()
    mesh = grid.partition(comm)
    logger.log(mesh, rank=-1)

if __name__ == "__main__":
    main()


