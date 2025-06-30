from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import init_process_group
from torch.distributed.pipeline.sync import Pipe
from torch.utils.data import DataLoader, TensorDataset

# Encoder e Decoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.seq(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.seq(x)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Setup torch.distributed
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Dummy dataset
    x = torch.rand(1000, 784)
    dataset = TensorDataset(x, x)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Costruzione del modello pipeline
    if rank == 0:
        model = nn.Sequential(Encoder(), Decoder())
        devices = [0, 1]
        model = Pipe(model, chunks=4, devices=devices).to(rank)
    else:
        model = None  # Solo rank 0 costruisce il modello completo

    if model is not None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(5):
            for data, target in dataloader:
                data = data.cuda(rank)
                target = target.cuda(rank)

                output = model(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"[Rank {rank}] Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()