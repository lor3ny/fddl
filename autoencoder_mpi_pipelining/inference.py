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
    img, _ = data.testloader.dataset[0]
    img_linear = img.view(-1).unsqueeze(0).to(device)

    model0_output = model0(img_linear)
    model1_output = model1(model0_output)
    model2_output = model2(model1_output)
    output = model3(model2_output)

    # Visualize original and reconstructed image
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img.squeeze(), cmap='gray')
    axs[0].set_title('Original')

    axs[1].imshow(output[0].view(28, 28).cpu(), cmap='gray')
    axs[1].set_title('Reconstructed')
    for ax in axs: ax.axis('off')
    plt.show()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')