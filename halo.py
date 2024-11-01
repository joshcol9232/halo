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


    def fetch_data_from(owning_partition, owned_data, global_idx):



    def exchange_owned_at_halo_locs(self, owned_data, comm=MPI.COMM_WORLD) -> np.array:
        """
        Assume that this is called on all PEs.

        Needs to be done sequentially. Do exchanges in order of rank.
        """

        local_halo_data_buffer = np.array_like(self.halo_indices)

        for rank in range(comm.size):
            if rank == comm.rank:
                # If my turn, retrieve data for each of my global indices.
                for local_idx, global_idx in enumerate(self.halo_indices):
                    owning_partition = self.table[halo_idx]

                    local_halo_data_buffer[local_idx] = fetch_data_from(owning_partition, owned_data, global_idx)
            else:
                self.serve()

        


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    grid = meshing.Grid3x3()
    mesh = grid.partition(comm)
    logger.log(mesh, rank=-1)

if __name__ == "__main__":
    main()


