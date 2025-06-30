import torch
import torch.nn as nn
import torch.optim as optim

from autoencoder import Autoencoder, Encoder, Decoder
from mnist_loader import MNISTLoader

data = MNISTLoader('./data/MNIST')

latent_linear_size: int = 3

#model = Autoencoder(data.linear_size, latent_linear_size)
modelA = Encoder()
modelB = Decoder()
criterion = nn.MSELoss()
optimizerA = optim.Adam(modelA.parameters(), lr=1e-3)
optimizerB = optim.Adam(modelB.parameters(), lr=1e-3)


device = torch.device('cuda')
if torch.cuda.is_available():
    print("Using GPU for training")
else:
    print("Using CPU for training")
    device = torch.device('cpu')
modelA.to(device)
modelB.to(device)

epochs = 10
for epoch in range(epochs):
    for images, _ in data.trainloader:
        inputs = images.view(-1, data.linear_size).to(device)

        # 1) Encoder forward
        latent = modelA(inputs)

        # detach so that encoder/decoder passes are truly separate
        latent = latent.detach().requires_grad_()

        # 2) Decoder forward + backward
        optimizerB.zero_grad()
        recon = modelB(latent)
        loss  = criterion(recon, inputs)
        loss.backward(retain_graph=True)

        # latent.grad now holds dLoss/d(latent)
        grad_latent = latent.grad

        # 3) Encoder backward
        optimizerA.zero_grad()
        latent.backward(grad_latent)   # push the gradient back
        optimizerA.step()
        optimizerB.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(modelA.state_dict(), 'encoder.pth')
torch.save(modelB.state_dict(), 'decoder.pth')
print("Models saved to encoder.pth and decoder.pth")