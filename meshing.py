# Module to help with testing the halo exchanges.

import numpy as np
import logger
from mpi4py import MPI

import halo

class Field:
    def __init__(self, mesh):
        self.mesh = mesh
        self.data = np.zeros(mesh.total_size())

    def data(self) -> np.array:
        return self.data

    def owned(self) -> np.array:
        return self.data[:self.mesh.len_owned()]

    def halo(self) -> np.array:
        return self.data[self.mesh.len_owned():]

    def halo_exchange(self):
        return

class Mesh:
    def __init__(self, owned, halo):
        self.owned_indices = owned
        self.halo_indices = halo

    def total_size(self) -> int:
        return len(self.owned_indices) + len(self.halo_indices)

    def len_owned(self) -> int:
        return len(self.owned_indices)

    def make_field(self) -> Field:
        return Field(self)


class Grid3x3:
    def __init__(self):
        self.indices = np.array([0, 1, 2,
                                 3, 4, 5,
                                 6, 7, 8])

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

    def partition(self, comm=MPI.COMM_WORLD) -> Mesh:
        # Available parition schemes:
        # "Individual" : each cell is an MPI rank. world size = 9.
        # "Stripes" : Each horizontal strip.
        num_ranks = comm.Get_size()

        owned_indices = []

        for global_idx, point in enumerate(self.partition_mapping(num_ranks)):
            if point == comm.rank:   # If that index is owned by this rank
                owned_indices.append(self.indices[global_idx])
        
        # Build the halo for the partition
        halo_indices = self.build_halo(owned_indices)
        return Mesh(owned_indices, halo_indices)

    def partition_mapping(self, world_size) -> np.array:
        """
        Gives a map of what global indices are owned by which processing element.
        """
        mapping = np.zeros_like(self.indices)
        mapping[:] = -1

        if world_size == 9:
            # In the case of 9 PEs, each cell is owned by it's associated rank
            mapping = np.copy(self.indices)
        else:
            raise ValueError("Grid3x3.partition_mapping: ERROR: Expected either 3 or 9 mpi ranks.")
        
        return mapping


