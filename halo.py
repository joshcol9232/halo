import numpy as np

import logger
from mpi4py import MPI

import meshing

class HaloTable:
    """
    On the local PE, we need a table of what halo points are owned by what rank.
    This can then be used for performing a halo exchange.
    """
    def __init__(self, partition_mapping, halo_indices):
        self.table = np.zeros_like(halo_indices)
        self.table[:] = -1
        
        # Partition mapping contains the global grid of what global index
        # corresponds to which rank it is owned by.
        for i, halo_idx in enumerate(halo_indices):
            self.table[i] = partition_mapping[halo_idx]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    grid = meshing.Grid3x3()
    mesh = grid.partition(comm)
    logger.log(mesh, rank=-1)

if __name__ == "__main__":
    main()


