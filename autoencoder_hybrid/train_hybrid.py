import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import numpy as np
import os
import argparse

from autoencoder import Autoencoder
from autoencoder import DistributedOperations as DOS
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
        comm: MPI.Comm
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
        self.device="cuda" #  #  Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu,


    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    # Actually we are using this trainer with an Autoencoder, so I don't really know if this evaluator is good
    '''
    def _evaluate (self) -> None:
        
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100 * correct / total
        
        return accuracy
    '''
        
    def _synchronize_gradients(self) -> None:

        for param in self.model.parameters():
            if param.grad is None:
                continue
            # Gather gradients from all ranks
            grad = param.grad.data.cpu().numpy()
            avg_grad = np.zeros_like(grad)
            try:
                self.comm.Allreduce(grad, avg_grad, op=MPI.SUM)
                avg_grad /= self.size  # Average the gradients
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
                print(f"-> Epoch {epoch} | Batch: {batch_idx}") if self.rank == 0 else None

                if self.rank == 0:
                    inputs = batch_data.view(-1, 28*28)
                    inputs = inputs.to(self.gpu_rank)
                    outputs = self.model(inputs)
                    loss = self.criterion(inputs, outputs)
                    
                    self.optimizer.zero_grad()
                    loss.backward()

                    sync_start_time = MPI.Wtime()
                    #self._synchronize_gradients() # VA FATTO SOLO SU 0
                    sync_end_time = MPI.Wtime()

                    local_syncs_lat.append(sync_end_time-sync_start_time)
                    total_loss += loss.item()

                    self.optimizer.step()
                else:
                    inputs = batch_data.view(-1, 28*28)
                    inputs = inputs.to(self.gpu_rank)
                    self.model(inputs)
            
            '''
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            '''

            local_epochs_lat.append(MPI.Wtime() - e_start_time)
            avg_loss = total_loss / len(self.train_data)
            all_avg_loss = (self.comm.allreduce(avg_loss, op=MPI.SUM) / self.size)
            print(f"-> Epoch {epoch} | Avg Loss: {all_avg_loss}") if self.rank == 0 else None

            #test_accuracy = self._evaluate()
            
        local_training_time = MPI.Wtime() - start_time
        max_training_time = self.comm.allreduce(local_training_time, op=MPI.MAX)

        print(f"Final execution time: {max_training_time}s") if self.rank == 0 else None
        
        # elif self.rank != 0:
        #     while True:
        #         # You can use tags or predefined commands
        #         status = MPI.Status()
        #         self.comm.probe(source=0, tag=MPI.ANY_TAG, status=status)
        #         if status.tag == 0:  # SHUTDOWN signal
        #             break
        #         elif status.tag == 1:  # e.g., MatMul request
        #             # Receive tensors, perform ops, send back
        #             DOS.DistributedMatmul(rank=self.rank, gpu_rank=self.gpu_rank, comm=self.comm)

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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    train_loader, test_loader = load_distribute_data(rank=rank, size=size, batch_size=batch_size, comm=comm)

    # MODEL INIT
    model = Autoencoder(rank, gpu_rank, comm, size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(gpu_rank)
    # not needed
    #model = DDP(model, device_ids=[gpu_rank])

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
        comm=comm
    )
    trainer.train(epochs)
    comm.Barrier()

    # # Il training deve gestire l'aggregazione del gradiente con la allreduce, rivederlo

    print(f"[RANK {rank}] Training done.", flush=True) if rank == 0 else None

    if(rank == 0):
        torch.save(model.state_dict(), "autoencoder_ddp.pth")
        print(f"[RANK: {rank}] Model saved to autoencoder_ddp.pth")


if __name__ == "__main__":

    epochs = 10
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