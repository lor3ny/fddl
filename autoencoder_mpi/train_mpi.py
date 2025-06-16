import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import os
import argparse

from autoencoder import Autoencoder
from mnist_loader import MNISTLoader


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data: MNISTLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        save_every: int,
        rank: int,
        gpu_rank: int,
        size: int,
        comm: MPI.Comm
    ) -> None:
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = model
        self.gpu_rank = gpu_rank
        self.rank = rank

    def _run_batch(self, inputs):
        outputs = self.model(inputs)
        loss = self.criterion(inputs, outputs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Steps: {len(self.data)}")
        for images, _ in self.data:
            inputs = images.view(-1, 28*28)
            inputs = inputs.to(self.local_rank)
            self._run_batch(inputs)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):

        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            '''
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            '''


# NON SONO CONVINTO VADA BENE, BISOGNA TESTARLO E CORREGGERLO

def load_distribute_data(
        rank: int, 
        size: int, 
        batch_size: int
    ) -> tuple[DataLoader, DataLoader]:


    if not os.path.exists("./data/MNIST"):
        if rank == 0:
            print(f"[RANK: {rank}] Downloading MNIST dataset...")
            datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print(f"[RANK: {rank}] DONE.")
    dist.barrier()

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    # Split dataset into subsets for each rank
    total_samples = len(train_dataset)
    sample_per_rank = total_samples // size
    remainder = total_samples % size
    indices = list(range(total_samples))

    # Distribute samples among ranks
    start_index = rank * sample_per_rank + min(rank, remainder)
    extra = 1 if rank < remainder else 0
    local_samples = sample_per_rank + extra
    end_index = start_index + local_samples
    local_indices = indices[start_index:end_index]
    print(f"[RANK: {rank}] I have {local_samples} samples. indices {start_index}:{end_index}")

    # Create DataLoader for local dataset
    # Bisogna passare local_indices
    local_dataset = set(train_dataset, local_indices)
    train_loader = DataLoader(
        local_dataset, batch_size=batch_size, shuffle=True
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
        print("x --- LIBRARY CHECK --- x")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}")
        print(f"GPUs per node {torch.cuda.device_count()}")
        print(f"Training with GPUs :)") if torch.cuda.is_available() else print("Training with CPUs")
        print("x --------------------- x")
    
    comm.barrier()
    print(f"[Rank {dist.get_rank()}] LOCAL_RANK={gpu_rank} on CUDA device {torch.cuda.current_device()} hostname={os.uname()[1]}")
    comm.barrier()

    # DATA MUST BE DIVIDED MANUALLY
    train_loader, test_loadr = load_distribute_data(rank=rank, size=size)

    # Verificare che questa funzione tiri fuori roba seria

    # MODEL INIT
    model = Autoencoder(28*28, 32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = model.to(gpu_rank)
    model = DDP(model, device_ids=[gpu_rank])

    # MODEL TRAINING
    print(f"[RANK: {rank}] Trainer is running...") if rank == 0 else None

    trainer = Trainer(model, train_loader, optimizer, criterion, save_every, rank, size, comm)
    trainer.train(epochs)
    comm.barrier()

    # Il training deve gestire l'aggregazione del gradiente con la allreduce, rivederlo

    print(f"[RANK: {rank}] Training done.") if rank == 0 else None

    if(rank == 0):
        torch.save(model.module.state_dict(), "autoencoder_ddp.pth")
        print(f"[RANK: {rank}] Model saved to autoencoder_ddp.pth")


if __name__ == "__main__":

    epochs = 1
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
