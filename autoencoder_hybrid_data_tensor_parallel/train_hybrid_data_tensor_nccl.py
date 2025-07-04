import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mpi4py import MPI
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import os
import argparse

# My API
from autoencoder import Autoencoder



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
        group_node: dist.ProcessGroup,
        group_0: dist.ProcessGroup
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
        self.group_node = group_node
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
            # Gather gradients from all ranks
            grad = param.grad.data.cpu().numpy()
            avg_grad = np.zeros_like(grad)
            try:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.group_0)
                avg_grad = grad / self.size  # Average the gradients
                param.grad.data = torch.tensor(avg_grad, device=self.device)

            except Exception as e:
                raise RuntimeError(f"Error synchronizing gradients: {e}.", flush=True)


    def train(self, max_epochs: int):

        local_syncs_lat = []
        local_epochs_lat = []

        start_time = MPI.Wtime()

        for epoch in range(max_epochs):
            print(f"[RANK {self.rank} GPU {self.gpu_rank}] Epoch {epoch} | Batches: {len(self.test_data)}") if self.rank == 0 else None
            e_start_time = MPI.Wtime()
            total_loss = 0.0

            for batch_idx, (batch_data, _) in enumerate(self.train_data):

                if self.gpu_rank == 0:
                    inputs = batch_data.view(-1, 28*28)
                    inputs = inputs.to(self.gpu_rank)

                    outputs = self.model(inputs)
                    loss = self.criterion(inputs, outputs)
                    
                    self.optimizer.zero_grad()
                    loss.backward()

                    sync_start_time = MPI.Wtime()
                    self._synchronize_gradients() # VA FATTO SOLO SU 0
                    sync_end_time = MPI.Wtime()

                    local_syncs_lat.append(sync_end_time-sync_start_time)
                    total_loss += loss.item()

                    self.optimizer.step()
                else:

                    for layers in range(2):

                        local_N = None
                        K = None
                        M = None

                        print(f"{self.rank} qui 0")

                        dist.broadcast(local_N, src=0, group=self.group_node)
                        dist.broadcast(K, src=0, group=self.group_node)
                        dist.broadcast(M, src=0, group=self.group_node)

                        print(f"{self.rank} qui 1")

                        # Scatter A over the ranks
                        A_local = torch.empty(local_N, K, dtype=torch.float32, device=self.gpu_rank)
                        weights_local = torch.empty(K, M, dtype=torch.float32, device=self.gpu_rank)
                        dist.scatter(tensor=A_local, scatter_list=None, src=0, group=self.group_node)
                        dist.broadcastcast(weights_local, src=0, group=self.group_node)

                        # # Bring everything to the GPU
                        # A_local = A_local.to(self.gpu_rank)
                        # weights_local = weights_local.to(self.gpu_rank)

                        print(f"{self.rank} qui 2")

                        dist.barrier()

                        print(f"{self.rank} qui 3")

                        # Actual matmul
                        C_local = A_local @ weights_local

                        #print(f"{self.rank} qui3")
                        dist.gather(C_local, gather_list=None, dst=0, group=self.group_node)

            if self.rank == 0:
                #local_epochs_lat.append(MPI.Wtime() - e_start_time)
                avg_loss = total_loss / len(self.train_data)
                #all_avg_loss = (self.comm.allreduce(avg_loss, op=MPI.SUM) / self.size)
                print(f"-> Epoch {epoch} | Avg Loss: {avg_loss}") if self.rank == 0 else None

            #test_accuracy = self._evaluate()
            
        training_time = MPI.Wtime() - start_time
        dist.all_reduce(training_time, op=MPI.MAX)

        print(f"Final execution time: {training_time}s") if self.rank == 0 else None
        
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

    # # Split dataset into subsets for each rank
    # total_samples = len(train_dataset)
    # if rank == 0:
    #     print(f"[RANK {rank}] This dataset has {total_samples} samples", flush=True)

    # sample_per_rank = total_samples // size
    # remainder = total_samples % size
    # indices = list(range(total_samples))

    # # Distribute samples among ranks
    # start_index = rank * sample_per_rank + min(rank, remainder)
    # extra = 1 if rank < remainder else 0
    # local_samples = sample_per_rank + extra
    # end_index = start_index + local_samples
    # local_indices = indices[start_index:end_index]
    # print(f"[RANK {rank}] I have {local_samples} samples. From {start_index} to {end_index}", flush=True)

    # # Create DataLoader for local dataset
    # # Bisogna passare local_indices
    # local_dataset = Subset(train_dataset, local_indices)
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
    print("here 0")
    local_rank = int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])
    backend = 'nccl'
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()

    print("here 1")

   # Group with specific global ranks
    included_ranks = [i for i in range(0, size, 4)]
    group_0 = dist.new_group(ranks=included_ranks)

    print("here 2")

    # Node-local group: group every 4 ranks
    my_node = rank // 4
    start_rank = my_node * 4
    node_ranks = list(range(start_rank, start_rank + 4))
    node_group = dist.new_group(ranks=node_ranks)

    gpu_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_rank)

    print("here 3")

    if rank == 0:
        print("+ ---------- TORCH LIBRARY CHECK ---------- +")
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"PyTorch CUDA version: {torch.version.cuda}", flush=True)
        print(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}", flush=True)
        print(f"GPUs per node {torch.cuda.device_count()}", flush=True)
        print(f"Training with GPUs :)") if torch.cuda.is_available() else print("Training with CPUs", flush=True)
        print("+ ----------------------------------------- +")
    
    dist.barrier()
    print(f"[Rank {rank}] GPU_RANK={gpu_rank} on CUDA device {torch.cuda.current_device()} hostname={os.uname()[1]}", flush=True)

    # DATA MUST BE DIVIDED MANUALLY
    dist.barrier()
    train_loader, test_loader = load_distribute_data(rank=rank, size=size, batch_size=batch_size, group=group_0)

    # MODEL INIT
    model = Autoencoder(rank, gpu_rank, node_group, size//2)
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
        save_every=save_every,
        rank=rank,
        gpu_rank=gpu_rank,
        size=size,
        group_node=node_group,
        group_0=group_0
    )
    trainer.train(epochs)
    dist.barrier()

    # # Il training deve gestire l'aggregazione del gradiente con la allreduce, rivederlo

    print(f"[RANK {rank}] Training done.", flush=True) if rank == 0 else None

    if(rank == 0):
        torch.save(model.state_dict(), "autoencoder_ddp.pth")
        print(f"[RANK: {rank}] Model saved to autoencoder_ddp.pth")

    dist.destroy_process_group()


if __name__ == "__main__":

    epochs = 10
    batch_size = 1024
    save_every = 1
    
    parser = argparse.ArgumentParser(description="Example of parsing many CLI arguments.")
    parser.add_argument("--ntasks", type=int, help="Number of tasks", default=1)
    args = parser.parse_args()
    world_size = args.ntasks

    main(
        epochs=epochs,
        batch_size=batch_size,
        save_every=save_every
    )