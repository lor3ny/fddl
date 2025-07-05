from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import csv

# My API
from autoencoder import Autoencoder


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
        comm: MPI.Comm,
        comm_0: MPI.Comm
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
        self.comm = comm
        self.comm_0 = comm_0
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
            # Gather gradients from all ranks
            grad = param.grad.data.cpu().numpy()
            avg_grad = np.zeros_like(grad)
            try:
                self.comm_0.Allreduce(grad, avg_grad, op=MPI.SUM)
                avg_grad /= self.size  # Average the gradients
                param.grad.data = torch.tensor(avg_grad, device=self.device)

            except Exception as e:
                raise RuntimeError(f"Error synchronizing gradients: {e}.", flush=True)

    def train(self, max_epochs: int):

        local_batch_lat = []

        start_time = MPI.Wtime()

        for epoch in range(max_epochs):
            print(f"[RANK {self.rank} GPU {self.gpu_rank}] Epoch {epoch} | Batches: {len(self.test_data)}") if self.rank == 0 else None
            total_loss = 0.0

            start_batch = MPI.Wtime()

            if self.gpu_rank == 0:
                for batch_idx, (batch_data, _) in enumerate(self.train_data):

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

            else:
                for batch_idx in range(self.batch_count):
                    for layer_idx in range(2):

                        # GPUs 1, 2 , 3 are slaves
                        local_N = None
                        K = None
                        M = None

                        local_N = self.comm.bcast(local_N, root = 0)
                        K = self.comm.bcast(K, root = 0)
                        M = self.comm.bcast(M, root = 0)

                        # Scatter A over the ranks
                        A_local = torch.zeros(int(local_N), int(K), dtype=torch.float32)
                        weights_local = torch.zeros(int(K), int(M), dtype=torch.float32)
                        self.comm.Scatter(None, [A_local, MPI.FLOAT], root=0)
                        self.comm.Bcast(weights_local, root=0)

                        # Bring everything to the GPU
                        A_local = A_local.to(self.gpu_rank)
                        weights_local = weights_local.to(self.gpu_rank)

                        self.comm.Barrier()

                        # Actual matmul
                        C_local = A_local @ weights_local
                        C_local_cpu = C_local.cpu()
                        self.comm.Gather([C_local_cpu, MPI.FLOAT], None, root=0)

            if self.gpu_rank == 0:
                local_batch_lat.append(MPI.Wtime() - start_batch)

            if self.rank == 0:
                avg_loss = total_loss / len(self.train_data)
                print(f"-> Epoch {epoch} | Avg Loss: {avg_loss}") if self.rank == 0 else None
            
        local_training_time = MPI.Wtime() - start_time
        max_training_time = self.comm.allreduce(local_training_time, op=MPI.MAX)

        print(f"Final execution time: {max_training_time}s") if self.rank == 0 else None

        if self.gpu_rank == 0:
            global_batch_lat = self.comm_0.allreduce(local_batch_lat, op=MPI.MAX)

        if self.rank == 0:
            SaveLatenciesCSV("Batch_Tensor_Parallel_MPI", global_batch_lat)
        
def load_distribute_data(
        rank: int, 
        size: int, 
        batch_size: int,
        comm: MPI.Comm
    ) -> tuple[DataLoader, DataLoader]:


    if not os.path.exists("./data/MNIST"):
        if rank == 0:
            print(f"[RANK: {rank}] Downloading MNIST dataset...")
            datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print(f"[RANK: {rank}] DONE.")
    comm.barrier()

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
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Os group communicator
    included_ranks = [i for i in range(0, size, 4)]
    world_group = comm.Get_group()
    group_0 = world_group.Incl(included_ranks)
    comm_0 = comm.Create(group_0)

    # Node group communicator 
    my_node = rank // 4
    start_rank = my_node * 4
    node_ranks = list(range(start_rank, start_rank + 4))
    world_group = comm.Get_group()
    node_group = world_group.Incl(node_ranks)
    node_comm = comm.Create(node_group)

    gpu_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_rank)

    if rank == 0:
        print("+ ---------- TORCH LIBRARY CHECK ---------- +")
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"PyTorch CUDA version: {torch.version.cuda}", flush=True)
        print(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}", flush=True)
        print(f"GPUs per node {torch.cuda.device_count()}", flush=True)
        print(f"Training with GPUs :)") if torch.cuda.is_available() else print("Training with CPUs", flush=True)
        print("+ ----------------------------------------- +")
    
    comm.Barrier()
    print(f"[Rank {rank}] GPU_RANK={gpu_rank} on CUDA device {torch.cuda.current_device()} hostname={os.uname()[1]}", flush=True)

    # DATA MUST BE DIVIDED MANUALLY
    comm.Barrier()
    if gpu_rank == 0:
        train_loader, test_loader = load_distribute_data(rank=rank, size=size//4, batch_size=batch_size, comm=comm_0)
        batch_count = len(train_loader)
        node_comm.bcast(batch_count, root=0)
    else:
        train_loader, test_loader = None, None
        batch_count = node_comm.bcast(None, root=0)

        
    # MODEL INIT
    model = Autoencoder(rank, gpu_rank, node_comm, size//(size/4))
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
        comm=node_comm,
        comm_0=comm_0
    )
    trainer.train(epochs)
    comm.Barrier()

    print(f"[RANK {rank}] Training done.", flush=True) if rank == 0 else None

    if(rank == 0):
        torch.save(model.state_dict(), "autoencoder_mpi.pth")
        print(f"[RANK: {rank}] Model saved to autoencoder_mpi.pth")


if __name__ == "__main__":

    epochs = 10
    batch_size = 128
    
    parser = argparse.ArgumentParser(description="Example of parsing many CLI arguments.")
    parser.add_argument("--ntasks", type=int, help="Number of tasks", default=1)
    args = parser.parse_args()
    world_size = args.ntasks

    main(
        epochs=epochs,
        batch_size=batch_size,
    )