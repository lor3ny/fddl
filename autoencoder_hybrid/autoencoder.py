import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI


class LinearTEST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias, rank, gpu_rank, size, comm):
        
        ctx.save_for_backward(A, B, bias)
        B = B.t()
        C = A @ B
        return C + bias


    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # print("-----BACK-----", flush=True)
        # print(grad_output.size(), flush=True)
        # print(weight.t().size(), flush=True)
        # print("-----BACK-----", flush=True)


        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None

class LinearMPI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias, rank, gpu_rank, size, comm):
        
        ctx.save_for_backward(A, B, bias)
        B = B.t()
        N = A.size(dim=0)
        K = B.size(dim=0)
        M = B.size(dim=1)

        # # if rank == 0:
        # #     print("-----FOR-----", flush=True)
        # #     print(A.size(), flush=True)
        # #     print(B.size(), flush=True)
        # #     print("-----FOR-----", flush=True)
        
        # local_rows = N // size

        # # Scatter A over the ranks
        # A_local = torch.empty(local_rows, K, dtype=torch.float32)
        # comm.Scatter([A.contiguous().detach().cpu(), MPI.FLOAT], [A_local, MPI.FLOAT], root=0)
        # # Give weights B to every process
        # comm.Bcast([B.contiguous().detach().cpu(), MPI.FLOAT], root=0)

        # # Bring everything to the GPU
        # A_local = A_local.to(gpu_rank)
        # B = B.to(gpu_rank)

        # comm.Barrier()

        # # Actual matmul
        # C_local = A_local @ B

        A_parts = A.chunk(4, dim=0)
    
        # Calcola le parti su GPU diverse
        if rank == 0:
            C_local = torch.matmul(A_parts[0].to(gpu_rank), B.to(gpu_rank))
        elif rank == 1:
            C_local = torch.matmul(A_parts[1].to(gpu_rank), B.to(gpu_rank))
        elif rank == 2:
            C_local = torch.matmul(A_parts[2].to(gpu_rank), B.to(gpu_rank))
        elif rank == 3:
            C_local = torch.matmul(A_parts[3].to(gpu_rank), B.to(gpu_rank))
    

        if rank == 0:
            
            C_local_cpu = C_local.cpu()
            C_total = torch.zeros(N, M, dtype=torch.float32)
            comm.Barrier()
            comm.Gather([C_local_cpu, MPI.FLOAT], [C_total, MPI.FLOAT], root=0)
            comm.Barrier()

            print("-----FOR-MPI-----", flush=True)
            print(C_total, flush=True)
            print("::::::::::::::::::")
            print((A@B), flush=True)
            print("-----FOR-BASE-----", flush=True)

            return C_total.to(gpu_rank) + bias.to(gpu_rank)
        else:
            C_local_cpu = C_local.cpu()
            comm.Barrier()
            comm.Gather([C_local_cpu, MPI.FLOAT], None, root=0)
            comm.Barrier()
            return torch.zeros(N, M).to(gpu_rank)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_output = grad_output.contiguous()

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # print("-----BACK-----", flush=True)
        # print(grad_output.size(), flush=True)
        # print(weight.t().size(), flush=True)
        # print("-----BACK-----", flush=True)


        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None

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
            return C.to(gpu_rank)
        else:
            C_local_cpu = C_local.detach().cpu()
            comm.Gather([C_local_cpu, MPI.FLOAT], None, root=0)
            return torch.zeros(N, M).to(gpu_rank)


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
        if self.size > 1:
            out = LinearMPI.apply(x, self.weight, self.bias, self.rank, self.gpu_rank, self.size, self.comm)
            return out
        else:
            return LinearTEST.apply(x, self.weight, self.bias, self.rank, self.gpu_rank, self.size, self.comm)#torch.matmul(x, self.weight.t()) + self.bias

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

    def forward(self, x, tag):
        encoded = self.encoder(x)
        #print(f"RANK {self.rank} GPU {self.gpu_rank} Finished Encoder!", flush=True)
        decoded = self.decoder(encoded)
        #print(f"RANK {self.rank} GPU {self.gpu_rank} Finished Decoder!", flush=True)
        return decoded
    
    # PIPELINING

    def encode(self, x):
        return self.encoder(x)



    
