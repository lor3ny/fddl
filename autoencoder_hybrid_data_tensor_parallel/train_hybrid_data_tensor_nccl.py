from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import argparse
import csv

# My API
from autoencoder_nccl import Autoencoder


def SaveLatenciesCSV(name, latencies):
    with open(f'{name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name])  # Header
        for number in latencies:
            writer.writerow([number])

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        batch_count: int,
        rank: int,
        gpu_rank: int,
        size: int,
        node_group: dist.ProcessGroup,
        group_0: dist.ProcessGroup
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_count = batch_count
        self.model = model
        self.gpu_rank = gpu_rank
        self.rank = rank
        self.size = size
        self.node_group = node_group
        self.group_0 = group_0
        self.device="cuda" #  #  Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu,


    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

        
    def _synchronize_gradients(self) -> None:

        for param in self.model.parameters():
            if param.grad is None:
                continue

            grad = param.grad.data.detach().clone()
            try:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.group_0)
                avg_grad = grad / self.size 
                param.grad.data = avg_grad

            except Exception as e:
                raise RuntimeError(f"Error synchronizing gradients: {e}.")

    def train(self, max_epochs: int):

        local_batch_lat = []

        start_time = time.perf_counter()

        for epoch in range(max_epochs):
            print(f"[RANK {self.rank} GPU {self.gpu_rank}] Epoch {epoch} | Batches: {len(self.test_data)}") if self.rank == 0 else None
            total_loss = 0.0

            if self.gpu_rank == 0:
                for batch_idx, (batch_data, _) in enumerate(self.train_data):

                    start_batch = time.perf_counter()

                    # GPU 0 as coordinator and slave
                    inputs = batch_data.view(-1, 28*28)
                    inputs = inputs.to(self.gpu_rank)

                    outputs = self.model(inputs)
                    loss = self.criterion(inputs, outputs)
                    
                    self.optimizer.zero_grad()
                    loss.backward()

                    self._synchronize_gradients() # Is done only on GPU 0s
    
                    total_loss += loss.item()

                    self.optimizer.step()

                    local_batch_lat.append(time.perf_counter()-start_batch)

            else:
                for batch_idx in range(self.batch_count):
                    for layer_idx in range(2):

                        # GPUs 1, 2 , 3 are slaves
                        local_N = torch.tensor(0, device=self.gpu_rank)
                        K = torch.tensor(0, device=self.gpu_rank)
                        M = torch.tensor(0, device=self.gpu_rank)

                        dist.broadcast(local_N, src=0, group=self.node_group)
                        dist.broadcast(K, src=0, group=self.node_group)
                        dist.broadcast(M, src=0, group=self.node_group)
                        local_N = local_N.item()
                        K = K.item()
                        M = M.item()

                        # Scatter A over the ranks
                        A_local = torch.empty(int(local_N), int(K), dtype=torch.float32, device=self.gpu_rank)
                        weights_local = torch.empty(int(K), int(M), dtype=torch.float32, device=self.gpu_rank)
                        dist.scatter(tensor=A_local, scatter_list=None, src=0, group=self.node_group)
                        dist.broadcast(weights_local, src=0, group=self.node_group)

                        # Bring everything to the GPU
                        A_local = A_local.to(self.gpu_rank)
                        weights_local = weights_local.to(self.gpu_rank)

                        dist.barrier()

                        # Actual matmul
                        C_local = A_local @ weights_local
                        dist.gather(C_local, gather_list=None, dst=0, group=self.node_group)

            if self.rank == 0:
                avg_loss = total_loss / len(self.train_data)
                print(f"-> Epoch {epoch} | Avg Loss: {avg_loss}") if self.rank == 0 else None
        
        local_training_time = torch.tensor((time.perf_counter() - start_time), device=self.gpu_rank)
        dist.all_reduce(local_training_time, op=dist.ReduceOp.MAX, group=self.node_group)

        print(f"Final execution time: {local_training_time}s") if self.rank == 0 else None

        if self.gpu_rank == 0:
            local_batch_lat = torch.tensor(local_batch_lat, device=self.gpu_rank)
            dist.all_reduce(local_batch_lat, op=dist.ReduceOp.MAX, group=self.group_0)
            local_batch_lat = local_batch_lat.tolist()
        
        if self.rank == 0:
            SaveLatenciesCSV("Batch_Tensor_Parallel_NCCL", local_batch_lat)
        
def load_distribute_data(
        rank: int, 
        size: int, 
        batch_size: int,
        group: dist.ProcessGroup
    ) -> tuple[DataLoader, DataLoader]:


    if not os.path.exists("./data/MNIST"):
        if rank == 0:
            print(f"[RANK: {rank}] Downloading MNIST dataset...")
            datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print(f"[RANK: {rank}] DONE.")
    dist.barrier(group=group)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    # Split dataset into subsets for each rank
    total_samples = len(train_dataset)
    if rank == 0:
        print(f"[RANK {rank}] This dataset has {total_samples} samples", flush=True)

    sample_per_rank = total_samples // size
    remainder = total_samples % size
    indices = list(range(total_samples))

    rank_index = rank//4
    # Distribute samples among ranks
    start_index = rank_index * sample_per_rank + min(rank_index, remainder)
    extra = 1 if rank_index < remainder else 0
    local_samples = sample_per_rank + extra
    end_index = start_index + local_samples
    local_indices = indices[int(start_index):int(end_index)]
    print(f"[RANK {rank}] I have {local_samples} samples. From {start_index} to {end_index}", flush=True)

    # Create DataLoader for local dataset
    local_dataset = Subset(train_dataset, local_indices)

    train_loader = DataLoader(
        local_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def main(
    epochs: int,
    batch_size: int
):
    # Initialize DIST for NCCL
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    backend = 'nccl'
    dist.init_process_group(backend=backend, init_method="env://", world_size=size, rank=rank)
    rank = dist.get_rank()
    gpu_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_rank)

    # Group with specific global ranks
    included_ranks = [i for i in range(0, size, 4)]
    group_0 = dist.new_group(ranks=included_ranks)

    # Node-local group: group every 4 ranks
    my_node = rank // 4
    start_rank = my_node * 4
    node_ranks = list(range(start_rank, start_rank + 4))
    node_group = dist.new_group(ranks=node_ranks)


    if rank == 0:
        print("+ ---------- TORCH LIBRARY CHECK ---------- +")
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"PyTorch CUDA version: {torch.version.cuda}", flush=True)
        print(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}", flush=True)
        print(f"GPUs per node {torch.cuda.device_count()}", flush=True)
        print(f"Training with GPUs :)") if torch.cuda.is_available() else print("Training with CPUs", flush=True)
        print("+ ----------------------------------------- +")
    
    print(f"[Rank {rank}] GPU_RANK={gpu_rank} on CUDA device {torch.cuda.current_device()} hostname={os.uname()[1]}", flush=True)

    # DATA MUST BE DIVIDED MANUALLY
    dist.barrier()
    if gpu_rank == 0:
        train_loader, test_loader = load_distribute_data(rank=rank, size=size//4, batch_size=batch_size, group=group_0)
        batch_count = torch.tensor(len(train_loader), dtype=torch.long).to(gpu_rank)
        dist.broadcast(batch_count, src=0, group=node_group)
        batch_count = batch_count.item()
    else:
        train_loader, test_loader = None, None
        batch_count=None
        batch_count = torch.tensor(0, dtype=torch.long, device=gpu_rank)
        dist.broadcast(batch_count, src=0, group=node_group)
        batch_count = batch_count.item()

    # MODEL INIT
    model = Autoencoder(rank, gpu_rank, node_group, size//(size/4))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(gpu_rank)

    # #MODEL TRAINING
    print(f"[RANK {rank}] Trainer is running...", flush=True) if rank == 0 else None

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        test_data=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        batch_count=batch_count,
        rank=rank,
        gpu_rank=gpu_rank,
        size=size,
        node_group=node_group,
        group_0=group_0
    )
    trainer.train(epochs)
    dist.barrier()

    print(f"[RANK {rank}] Training done.", flush=True) if rank == 0 else None

    if(rank == 0):
        torch.save(model.state_dict(), "autoencoder_nccl.pth")
        print(f"[RANK {rank}] Model saved to autoencoder_nccl.pth")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    epochs = 20
    batch_size = 1024
    
    parser = argparse.ArgumentParser(description="Example of parsing many CLI arguments.")
    parser.add_argument("--ntasks", type=int, help="Number of tasks", default=1)
    args = parser.parse_args()
    world_size = args.ntasks

    main(
        epochs=epochs,
        batch_size=batch_size,
    )