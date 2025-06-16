from mpi4py import MPI
import os


if __name__ == "__main__":

     # Initialize MPI

     print(MPI.get_vendor())

     comm = MPI.COMM_WORLD
     rank = comm.Get_rank()
     size = comm.Get_size()

     gpu_rank = rank % 4
    
     comm.barrier()
     print(f"[Rank {rank}] LOCAL_RANK={gpu_rank} on CUDA device {gpu_rank} hostname={os.uname()[1]}")
     comm.barrier()
