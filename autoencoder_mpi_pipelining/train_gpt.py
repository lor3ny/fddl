from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size == 2, "This example requires exactly 2 MPI ranks."

device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

# Define model parts
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Hyperparameters
epochs = 10
batch_size = 64
input_dim = 784
hidden_dim = 256
learning_rate = 1e-3

# Create dummy dataset (e.g., MNIST flattened)

transform = transforms.ToTensor()
if not os.path.exists("./data/MNIST"):
    if rank == 0:
        print(f"[RANK: {rank}] Downloading MNIST dataset...")
        datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        print(f"[RANK: {rank}] DONE.")
comm.barrier()


train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate model parts and optimizers on respective GPUs
if rank == 0:
    model = Encoder(input_dim, hidden_dim).to(device0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.view(-1, 28*28).to(device0)
            # Forward on encoder
            hidden = model(data)
            # Send hidden activations to decoder rank
            comm.send(hidden.cpu(), dest=1, tag=11)

            # Receive gradient for hidden activations from decoder
            hidden_grad = comm.recv(source=1, tag=22)
            hidden_grad = hidden_grad.to(device0)

            # Backward on encoder
            model.zero_grad()
            hidden.backward(hidden_grad)
            optimizer.step()

        print(f"Rank 0 | Epoch {epoch} | Batch {batch_idx}")

elif rank == 1:
    model = Decoder(hidden_dim, input_dim).to(device1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.view(-1, 28*28).to(device0)
            # Receive hidden activations from encoder rank
            hidden = comm.recv(source=0, tag=11)
            hidden = hidden.to(device1).requires_grad_()
            hidden.retain_grad()

            # Forward on decoder
            recon = model(hidden)
            loss = criterion(recon, data.to(device1))

            # Backward on decoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Send gradient of hidden back to encoder rank
            hidden_grad = hidden.grad.cpu()
            comm.send(hidden_grad, dest=0, tag=22)

        print(f"Rank 1 | Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")