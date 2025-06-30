# inference.py (or a new cell/file)
import torch
import matplotlib.pyplot as plt

from autoencoder import Autoencoder, Autoencoder_PIPE, Encoder, Decoder 
from mnist_loader import MNISTLoader


# Load model architecture
modelEncoder = Encoder()
modelDecoder = Decoder()
data = MNISTLoader('./data/MNIST')

# Load saved weights
modelEncoder.load_state_dict(torch.load("encoder.pth"))
modelDecoder.load_state_dict(torch.load("decoder.pth"))
modelEncoder.eval()  # Set to evaluation mode
modelDecoder.eval()  # Set to evaluation mode

device = torch.device('cuda')
if torch.cuda.is_available():
    print("Using GPU for training")
else:
    print("Using CPU for training")
    device = torch.device('cpu')
modelEncoder.to(device)
modelDecoder.to(device)

with torch.no_grad():
    img, _ = data.testloader.dataset[0]
    img_linear = img.view(-1).unsqueeze(0).to(device)

    outA = modelEncoder(img_linear)
    output = modelDecoder(outA)

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