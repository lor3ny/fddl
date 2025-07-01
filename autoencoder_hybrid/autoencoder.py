import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI




class DistributedOperations():
    def DistributedMatmul(rank, gpu_rank, size, comm, A=None, B=None):

        N = A.size(dim=0)
        K = B.size(dim=0)
        M = B.size(dim=1)
        
        local_rows = N // size

        # I think that I'm losing the gradient here, verify printing the gradient.
        # Have to exists a way to maintain the gradient here.

        # Copy() doesn't exist, how can I do?
        A_cp = A.copy()
        B_cp = B.copy()

        # Scatter A over the ranks
        A_local = torch.empty(local_rows, K, dtype=torch.float32)
        comm.Scatter([A_cp.detach().cpu(), MPI.FLOAT], [A_local, MPI.FLOAT], root=0)
        # Give weights B to every process
        comm.Bcast([B_cp.detach().cpu(), MPI.FLOAT], root=0)

        # Bring everything to the GPU
        A_local = A_local.to(gpu_rank)
        B = B.to(gpu_rank)

        # Actual matmul
        C_local = torch.matmul(A_local, B)

        if rank == 0:
            C_local_cpu = C_local.detach().cpu()
            C = torch.empty(N, M, dtype=torch.float32)
            comm.Gather([C_local_cpu, MPI.FLOAT], [C, MPI.FLOAT], root=0)
            C.grad = A.grad
            return C.to(gpu_rank)
        else:
            C_local_cpu = C_local.detach().cpu()
            comm.Gather([C_local_cpu, MPI.FLOAT], None, root=0)
            return torch.zeros(N, M, device=f"cuda:{gpu_rank}") 

class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, gpu_rank, comm, size):
        super(ManualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.gpu_rank = gpu_rank
        self.comm = comm
        self.size = size

        # Inizializzazione dei pesi e bias (con parametri ottimizzabili)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        # bias: (out_features)
        #print(f"RANK {self.rank} GPU {self.gpu_rank} Started MATMUL!", flush=True)
        if True:
            if self.rank == 0:
                return DistributedOperations.DistributedMatmul(self.rank, self.gpu_rank, self.size, self.comm, A=x, B=self.weight.t()) + self.bias #torch.matmul(x, self.weight.t()) + self.bias
            else:
                return DistributedOperations.DistributedMatmul(self.rank, self.gpu_rank, self.size, self.comm, A=x, B=self.weight.t())
        else:
            return torch.matmul(x, self.weight.t()) + self.bias

class Autoencoder(nn.Module):
    def __init__(self, rank=None, gpu_rank=None, comm=None, size=None):
        super(Autoencoder, self).__init__()
        self.rank = rank
        self.gpu_rank = gpu_rank
        self.comm = comm
        self.size = size

        self.encoder = nn.Sequential(
            ManualLinear(28*28, 256, rank, gpu_rank, comm, size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            ManualLinear(256, 28*28, rank, gpu_rank, comm, size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        #print(f"RANK {self.rank} GPU {self.gpu_rank} Finished Encoder!", flush=True)
        decoded = self.decoder(encoded)
        #print(f"RANK {self.rank} GPU {self.gpu_rank} Finished Decoder!", flush=True)
        return decoded
    
    # PIPELINING

    def encode(self, x):
        return self.encoder(x)



    
