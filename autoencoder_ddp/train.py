import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from autoencoder import Autoencoder
from mnist_loader import MNISTLoader


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os

def ddp_setup(rank, world_size):
    """Initialize the distributed environment."""    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup():
    """Clean up the distributed environment."""
    destroy_process_group()

'''
data = MNISTLoader('./data/MNIST')

latent_linear_size: int = 3

model = Autoencoder(data.linear_size, latent_linear_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


device = torch.device('cuda')
if torch.cuda.is_available():
    print("Using GPU for training")
else:
    print("Using CPU for training")
    device = torch.device('cpu')
model.to(device)


epochs = 5
for epoch in range(epochs):
    for images, _ in data.trainloader:
        inputs = images.view(-1, data.linear_size).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'autoencoder.pth')
print("Model saved to autoencoder.pth")
'''

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        data: MNISTLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        #self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, inputs):
        outputs = self.model(inputs)
        loss = self.criterion(inputs, outputs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        for images, _ in self.data.trainloader:
            inputs = images.view(-1, self.data.linear_size).to(self.gpu_id)
            self._run_batch(inputs)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def save_model(self, path: str):
        ckp = self.model.state_dict()
        torch.save(ckp, path)
        print(f"Model saved at {path}")

    def train(self, max_epochs: int):

        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            '''
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            '''


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, latent_linear_size: int):
    #ddp_setup(rank, world_size)

    dataset = MNISTLoader('./data/MNIST', batch_size=batch_size)
    model = Autoencoder(dataset.linear_size, latent_linear_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, dataset, optimizer, criterion, rank, save_every)

    trainer.train(total_epochs)
    if(rank == 0):
        trainer.save_model(f'autoencoder_master.pth')

    #ddp_cleanup()


if __name__ == "__main__":
    epochs = 5
    batch_size = 64
    save_every = 1
    latent_linear_size = 32

    world_size = torch.cuda.device_count()
    print(world_size, "GPUs available for training")
    main(0, world_size, save_every, epochs, batch_size, latent_linear_size)

    #mp.spawn(main, args=(world_size, save_every, epochs, batch_size, latent_linear_size), nprocs=world_size)