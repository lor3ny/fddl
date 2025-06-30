# inference.py (or a new cell/file)
import torch
import matplotlib.pyplot as plt

from autoencoder import Autoencoder, Autoencoder_PIPE, Layer0, Layer1, Layer2, Layer3
from mnist_loader import MNISTLoader


# Load model architecture
model =  Autoencoder_PIPE(28*28, 32)
data = MNISTLoader('./data/MNIST')


model0 = Layer0()
model1 = Layer1()
model2 = Layer2()
model3 = Layer3()

model0.load_state_dict(torch.load("layer0.pth", map_location="cuda:0"))
model1.load_state_dict(torch.load("layer1.pth", map_location="cuda:0"))
model2.load_state_dict(torch.load("layer2.pth", map_location="cuda:0"))
model3.load_state_dict(torch.load("layer3.pth", map_location="cuda:0"))

print(next(iter(model0.state_dict().values())).view(-1)[:10])
print(next(iter(model1.state_dict().values())).view(-1)[:10])
print(next(iter(model2.state_dict().values())).view(-1)[:10])
print(next(iter(model3.state_dict().values())).view(-1)[:10])

model0.eval()  # Set to evaluation mode
model1.eval()  # Set to evaluation mode
model2.eval()  # Set to evaluation mode
model3.eval()  # Set to evaluation mode

device = torch.device('cuda')
if torch.cuda.is_available():
    print("Using GPU for training")
else:
    print("Using CPU for training")
    device = torch.device('cpu')

model0 = model0.to(device)
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)


with torch.no_grad():
    for i in range(5):
        img, _ = data.testloader.dataset[i]
        img_linear = img.view(-1).unsqueeze(0).to(device)

        model0_output = model0(img_linear)
        model1_output = model1(model0_output)
        model2_output = model2(model1_output)
        output = model3(model2_output)

        # Visualize original and reconstructed image
        fig, axs = plt.subplots(1, 5)
        axs[0].imshow(img.squeeze(), cmap='gray')
        axs[0].set_title('Original')

        axs[1].imshow(model0_output[0].view(16, 16).cpu(), cmap='gray')
        axs[1].set_title('Encoded 0')

        axs[2].imshow(model1_output[0].view(8, 8).cpu(), cmap='gray')
        axs[2].set_title('Encoded 1')

        axs[3].imshow(model2_output[0].view(16, 16).cpu(), cmap='gray')
        axs[3].set_title('Decoded 0')

        axs[4].imshow(output[0].view(28, 28).cpu(), cmap='gray')
        axs[4].set_title('Decoded 1: Reconstructed')
        for ax in axs: ax.axis('off')
        plt.show()

plt.figure(figsize=(16, 32))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title("Encoded 0")
plt.axis('off')

plt.subplot(1, 2, 3)
plt.imshow(output, cmap='gray')
plt.title("Encoded 1")
plt.axis('off')

plt.subplot(1, 2, 4)
plt.imshow(output, cmap='gray')
plt.title("Decoded 0")
plt.axis('off')

plt.subplot(1, 2, 5)
plt.imshow(output, cmap='gray')
plt.title("Decoded 1: Reconstructed Image")
plt.axis('off')