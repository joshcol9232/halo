from mpi4py import MPI

def log(message, rank=0, comm=MPI.COMM_WORLD):
    if rank == -1 or comm.rank == rank:
        print(f"[PE {comm.rank}] {message}")


