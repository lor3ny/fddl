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
        with_gpu: bool
    ) -> None:
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.with_gpu = with_gpu
        self.model = model
        self.local_rank = int(os.environ["RANK"] % torch.cuda.device_count())
        self.global_rank = int(os.environ["RANK"])
        # IF YOU WANT TO ADD GPU SUPPORT YOU NEED TO KNOW HOW MANY GPUS YOU HAVE PER NODE
        # self.gpu_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else None

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
            if self.with_gpu:
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


def main_train(save_every: int, total_epochs: int, batch_size: int, latent_linear_size: int):


    print(f"[RANK {rank}] Hi! My local rank is {local_rank}, global rank is {rank}, world size is {world_size}")
    dist.barrier()

    if torch.cuda.is_available():
        print("Training with GPUs") if rank == 0 else None

    else:
        print("Training with CPUs") if rank == 0 else None



    # INITIALIZATION
    dist.init_process_group(backend='gloo')
    local_rank = int(os.environ["LOCAL_RANK"] % torch.cuda.device_count())
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])


    # DATASET LOADING
    if not os.path.exists("./data/MNIST"):
        if rank == 0:
            print(f"[RANK: {rank}] Downloading MNIST dataset...")
            datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print(f"[RANK: {rank}] DONE.")
    dist.barrier()

    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=False, sampler=DistributedSampler(trainset))
    testloader = DataLoader(testset, batch_size=64, shuffle=False, sampler=DistributedSampler(testset))


    print(f"[RANK {rank}] Current working directory: {os.getcwd()}")

    model = Autoencoder(28*28, latent_linear_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
        with_gpu = True
    else:
        model = DDP(model)  # CPU
        with_gpu = False

    # Teoricamente ora ogni core ha un batch di dati diverso, e fanno allreduce automaticamente
    # 1. Se fosse multinodo come facciamo? Lanciandolo su slurm funziona comunque?
        # Bisogna creare un env e attivarlo su slurm con conda, creare un file nevirnment.yaml
    # 2. Ora bisogna fare pipelining, quindi batch per nodo e pipelining per core o gpu
    # 3. Come faccio a fare batch per nodo su slurm e pytorch?

    print(f"[RANK: {rank}] Trainer is running...") if rank == 0 else None

    trainer = Trainer(model, trainloader, optimizer, criterion, save_every, with_gpu=with_gpu)
    trainer.train(total_epochs)
    dist.barrier()

    print(f"[RANK: {rank}] Training done.") if rank == 0 else None

    if(rank == 0):
        torch.save(model.module.state_dict(), "autoencoder_ddp.pth")
        print("[RANK: {rank}] Model saved to autoencoder_ddp.pth")

    dist.destroy_process_group()


if __name__ == "__main__":

    epochs = 1
    batch_size = 64
    save_every = 1
    latent_linear_size = 32
    
    parser = argparse.ArgumentParser(description="Example of parsing many CLI arguments.")
    parser.add_argument("--ntasks", type=int, help="Number of tasks", default=1)
    args = parser.parse_args()
    world_size = args.ntasks

    main_train(
        save_every=save_every,
        total_epochs=epochs,
        batch_size=batch_size,
        latent_linear_size=latent_linear_size
    )
