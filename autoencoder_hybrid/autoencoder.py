import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI




class DistributedOperations():
    def DistributedMatmul(rank, gpu_rank, size, comm, A=None, B=None):

        #comm.Ibcast(None, root=0, tag=2)
        
        # if rank == 0:
        #     N = A.size(dim=0)
        #     K = B.size(dim=0)
        #     M = B.size(dim=1)
        #     comm.Bcast([N, MPI.FLOAT], root=0, tag=2)
        #     comm.Bcast([K, MPI.FLOAT], root=0, tag=2)
        #     comm.Bcast([M, MPI.FLOAT], root=0, tag=2)
        # else:
        #     N = comm.Bcast([N, MPI.FLOAT], root=0, tag=2)
        #     K = comm.Bcast([K, MPI.FLOAT], root=0, tag=2)
        #     M = comm.Bcast([M, MPI.FLOAT], root=0, tag=2)

        N = A.size(dim=0)
        K = B.size(dim=0)
        M = B.size(dim=1)
        
        local_rows = N // size

        A_local = torch.empty(local_rows, K, dtype=torch.float32)
        comm.Scatter([A, MPI.FLOAT], [A_local, MPI.FLOAT], root=0)

        comm.Bcast([B, MPI.FLOAT], root=0)

        A_local = A_local.to()
        B = B.to(gpu_rank)

        C_local = torch.matmul(A_local, B)

        C_local_cpu = C_local.cpu()

        comm.Gather([C_local_cpu, MPI.FLOAT], [C, MPI.FLOAT], root=0)

        if rank == 0:
            C = torch.empty(N, M, dtype=torch.float32)
        else:
            C = None

        return C

class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ManualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Inizializzazione dei pesi e bias (con parametri ottimizzabili)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, rank, gpu_rank, comm):
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        # bias: (out_features)
        return DistributedOperations.DistributedMatmul(rank, gpu_rank, comm, A=x, B=self.weight.t()) + self.bias #torch.matmul(x, self.weight.t()) + self.bias
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ManualLinear(28*28, 256),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            ManualLinear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # PIPELINING

    def encode(self, x):
        return self.encoder(x)



    
