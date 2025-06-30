import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import torch.distributed as dist

from mpi4py import MPI
import numpy as np
import os
import argparse
import time
import sys

from autoencoder import Autoencoder_PIPE, Encoder, Decoder
#from mnist_loader import MNISTLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        save_every: int,
        rank: int,
        gpu_rank: int,
        size: int,
        comm: MPI.Comm,
        sync_comm: MPI.Comm
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = model
        self.gpu_rank = gpu_rank
        self.rank = rank
        self.size = size
        self.comm = comm
        self.sync_comm = sync_comm  # Used for synchronizing gradients
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
            grad = param.grad.data.cpu().numpy() # why cpu?
            avg_grad = np.zeros_like(grad)
            try:
                self.sync_comm.Allreduce(grad, avg_grad, op=MPI.SUM)
                avg_grad /= self.size  # Average the gradients
                param.grad.data = torch.tensor(avg_grad, device=self.device)

            except Exception as e:
                raise RuntimeError(f"Error synchronizing gradients: {e}.", flush=True)
            


    def train(self, max_epochs: int):
        local_syncs_lat = []
        local_epochs_lat = []

        dist.init_process_group('gloo', rank=self.rank, world_size=self.size)

        start_time = MPI.Wtime()
        for epoch in range(max_epochs):
            print(f"[RANK {self.rank} GPU {self.gpu_rank}] Epoch {epoch} | Batches: {len(self.test_data)}", flush=True) if self.rank == 0 else None
            e_start_time = MPI.Wtime()

            self.comm.Barrier()  # Synchronize all ranks before starting the epoch

            print(f"{self.rank} - {sum(p.numel() for p in self.model.parameters())}", flush=True) # Ensure model parameters are initialized

            for batch_idx, (batch_data, batch_target) in enumerate(self.train_data):

                #activations_storage = [None] * 1 # To store activations for backward pass
                #gradients_storage = [None] * 1 # To store gradients for backward pass

                 # --- Forward Pass ---

                if self.rank == 0: # First stage
                    # Compute activations for stage 0
                    inputs = batch_data.view(-1, 28*28).to(self.gpu_rank).requires_grad_()
                    activations = self.model(inputs.to(self.gpu_rank))

                    # Send activations to the next stage (rank 1)
                    dist.send(activations.cpu(), dest=1, tag=batch_idx)

                    #grad_received = torch.empty((len(batch_data), 256), dtype=torch.float32)
                    dist.recv(grad_received, source=1, tag=batch_idx)
                    grad_received = grad_received.to(self.gpu_rank)
                    self.optimizer.zero_grad()  # Reset gradients before backward pass
                    activations.backward(gradient=grad_received)

                elif self.rank == 1: # Last stage
                    # Receive activations from the previous stage (rank 0)
                    #received_activations = torch.empty((len(batch_data), 256), dtype=torch.float32)
                    dist.recv(received_activations, source=0, tag=batch_idx)
                    received_activations = received_activations.to(self.gpu_rank).requires_grad_()
                    #received_activations.retain_grad()
                    
                    # Compute activations for stage 1 (final output)
                    self.optimizer.zero_grad()  # Reset gradients before forward pass
                    outputs = self.model(received_activations)
                    inputs = batch_data.view(-1, 28*28).to(self.gpu_rank)
                    loss = self.criterion(outputs, inputs)
                    loss.backward(retain_graph=True)

                    latent_grad = received_activations.grad.cpu()
                    dist.send(latent_grad, dest=0, tag=batch_idx)

                self.optimizer.step()

            dist.Barrier()
            if self.gpu_rank == 1:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}', flush=True)




#     # NODES SYNCHRONIZATION
#     # sync_start_time = MPI.Wtime()
#     # print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} SYNCHING START", flush=True)
#     # self._synchronize_gradients() # in questo caso solo tra le gpu 3
#     # print(f"{self.rank} GPU {self.gpu_rank} -> Epoch {epoch} | Batch: {batch_idx} SYNCHING END", flush=True)
#     # sync_end_time = MPI.Wtime()
#     # local_syncs_lat.append(sync_end_time-sync_start_time)
#     # req.Wait() #non va messa qui




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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def main(
    epochs: int,
    batch_size: int,
    save_every: int,
):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    group_ranks = [0]
    world_group = comm.Get_group()
    sub_group = world_group.Incl(group_ranks)
    sub_comm = comm.Create(sub_group)

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

    # if rank == 0:
    train_loader, test_loader = load_distribute_data(rank=rank, size=size, batch_size=batch_size, comm=comm)

    # #MODEL TRAINING
    print(f"[RANK {rank}] Trainer is running...", flush=True) if rank == 0 else None

    layers = [Encoder, Decoder]
    model = layers[rank]().to(gpu_rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # # Il training deve gestire l'aggregazione del gradiente con la allreduce, rivederlo

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        test_data=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        save_every=save_every,
        rank=rank,
        gpu_rank=gpu_rank,
        size=size,
        comm=comm,
        sync_comm=sub_comm if rank in group_ranks else MPI.COMM_NULL
    )
    trainer.train(epochs)

    print(f"[RANK {rank}] Training done.", flush=True) if rank == 0 else None

    #if(rank == 0):
    torch.save(model.state_dict(), f"layer{gpu_rank}.pth")
    print(f"[RANK: {rank}] Model saved to layer{gpu_rank}.pth")


if __name__ == "__main__":

    epochs = 20
    batch_size = 64
    save_every = 1
    latent_linear_size = 32
    
    parser = argparse.ArgumentParser(description="Example of parsing many CLI arguments.")
    parser.add_argument("--ntasks", type=int, help="Number of tasks", default=1)
    args = parser.parse_args()
    world_size = args.ntasks


    main(
        epochs=epochs,
        batch_size=batch_size,
        save_every=save_every
    )
