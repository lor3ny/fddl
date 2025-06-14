import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from autoencoder import Autoencoder
from mnist_loader import MNISTLoader


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os

def ddp_setup(rank, world_size):
    """Initialize the distributed environment."""    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        init_process_group(backend='gloo', rank=rank, world_size=world_size)



def ddp_cleanup():
    """Clean up the distributed environment."""
    destroy_process_group()



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data: MNISTLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        rank: int,
        save_every: int,
        with_gpu: bool
    ) -> None:
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.with_gpu = with_gpu
        self.model = model
        self.rank = rank
        # IF YOU WANT TO ADD GPU SUPPORT YOU NEED TO KNOW HOW MANY GPUS YOU HAVE PER NODE
        # self.gpu_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else None

    def _run_batch(self, inputs):
        outputs = self.model(inputs)
        loss = self.criterion(inputs, outputs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        for images, _ in self.data.trainloader:
            inputs = images.view(-1, self.data.linear_size)
            if self.with_gpu:
                inputs = inputs.to(self.gpu_id)

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


def main_train(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, latent_linear_size: int):

    ddp_setup(rank, world_size)

    dataset = MNISTLoader('./data/MNIST', batch_size=batch_size)
    sampler = DistributedSampler(dataset.trainloader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataset.trainloader = DataLoader(dataset.trainloader.dataset, batch_size=batch_size, sampler=sampler)
    
    model = Autoencoder(dataset.linear_size, latent_linear_size)
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
    # 2. Ora bisogna fare pipelining, quindi batch per nodo e pipelining per core o gpu
    # 3. Come faccio a fare batch per nodo su slurm e pytorch?

    trainer = Trainer(model, dataset, optimizer, criterion, rank, save_every, with_gpu=with_gpu)
    trainer.train(total_epochs)


    if(rank == 0):
        torch.save(model.module.state_dict(), "autoencoder_ddp.pth")
        print("Model saved to autoencoder_ddp.pth")

    ddp_cleanup()


if __name__ == "__main__":

    epochs = 5
    batch_size = 64
    save_every = 1
    latent_linear_size = 32
    cuda = True

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        world_size = torch.cuda.device_count()
        print(world_size, "GPUs available for training")
    else:
        world_size = torch.get_num_threads()
        print(world_size, "COREs available for training")

    mp.spawn(main_train, args=(world_size, save_every, epochs, batch_size, latent_linear_size), nprocs=world_size, join=True)