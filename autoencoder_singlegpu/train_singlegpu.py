import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI
import csv

from autoencoder import Autoencoder
from mnist_loader import MNISTLoader

def SaveLatenciesCSV(name, latencies):
    with open(f'{name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name])  # Header
        for number in latencies:
            writer.writerow([number])


data = MNISTLoader('./data/MNIST')

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda')
if torch.cuda.is_available():
    print("Using GPU for training")
else:
    print("Using CPU for training")
    device = torch.device('cpu')
model.to(device)

epochs = 10
batch_lat = []
for epoch in range(epochs):
    for images, _ in data.trainloader:

        start_time = MPI.Wtime()

        inputs = images.view(-1, data.linear_size).to(device)
        # 1) Encoder forward
        output = model(inputs)

        # 2) Decoder forward + backward
        optimizer.zero_grad()
        loss = criterion(output, inputs)
        loss.backward()
        optimizer.step()

        batch_lat.append(MPI.Wtime() - start_time)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


SaveLatenciesCSV("Single GPU", batch_lat)

torch.save(model.state_dict(), 'autoencoder.pth')
print("Model saved to autoencoder.pth")