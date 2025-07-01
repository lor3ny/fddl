import torch
import torch.nn as nn
import torch.optim as optim

from autoencoder import Autoencoder
from mnist_loader import MNISTLoader

data = MNISTLoader('./data/MNIST')

latent_linear_size: int = 3

#model = Autoencoder(data.linear_size, latent_linear_size)
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
for epoch in range(epochs):
    for images, _ in data.trainloader:
        inputs = images.view(-1, data.linear_size).to(device)

        # 1) Encoder forward
        output = model(inputs)

        # 2) Decoder forward + backward
        optimizer.zero_grad()
        loss = criterion(output, inputs)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'autoencoder.pth')
print("Model saved to autoencoder.pth")