import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist



class LinearOverrideTEST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias):

        ctx.save_for_backward(A, B, bias)
        B = B.t()
        C = A @ B
        return C + bias

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None

class LinearMPI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias, rank, gpu_rank, size, group):
        
        ctx.save_for_backward(A, B, bias)
        B = B.t()
        N = A.size(dim=0)
        K = B.size(dim=0)
        M = B.size(dim=1)

        local_rows = int(N // size)

        dist.broadcast(local_rows, src=0, group=group)
        dist.broadcast(K, src=0, group=group)
        dist.broadcast(M, src=0, group=group)

        A_parts = A.chunk(4, dim=0)
        A_local = torch.empty(int(local_rows), int(K), dtype=torch.float32, device=gpu_rank)
        dist.scatter(tensor=A_local, scatter_list=A_parts, src=0, group=group)
        dist.broadcast(tensor=B, src=0, group=group)

        dist.barrier()

        # Actual matmul
        C_local = A_local @ B

        C_total = torch.empty(N, M, dtype=torch.float32)

        C_locals = [torch.empty(local_rows, M, dtype=torch.float32, device=gpu_rank) for _ in range(size//4)]
        dist.gather(C_local, gather_list=C_locals, dst=0, group=group)

        C_total = torch.cat([
            C_locals[0],
            C_locals[1],
            C_locals[2],
            C_locals[3]
        ], dim=0)

        # print(f"RESULT: {C_total.to(gpu_rank)}", flush=True)
        # print(f"TEST: {(A @ B)}", flush=True)
        return C_total.to(gpu_rank) + bias.to(gpu_rank)


    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_output = grad_output.contiguous()

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, gpu_rank, group, size):
        super(ManualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.gpu_rank = gpu_rank
        self.group = group
        self.size = size
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        if self.size > 1:
            out = LinearMPI.apply(x, self.weight, self.bias, self.rank, self.gpu_rank, self.size, self.group)
            return out
        else:
            return LinearOverrideTEST.apply(x, self.weight, self.bias)

class Autoencoder(nn.Module):
    def __init__(self, rank=None, gpu_rank=None, group=None, size=None):
        super(Autoencoder, self).__init__()
        self.rank = rank
        self.gpu_rank = gpu_rank
        self.group = group
        self.size = size

        self.encoder = nn.Sequential(
            ManualLinear(28*28, 256, rank, gpu_rank, group, size),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            ManualLinear(256, 28*28, rank, gpu_rank, group, size),
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



    
