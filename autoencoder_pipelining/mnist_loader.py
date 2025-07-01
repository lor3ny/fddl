from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


class MNISTLoader:
    def __init__(self, data_source, batch_size=64):
        self.data_source = data_source
        self.linear_size = 28*28 # MNIST image size
        self.batch_size = batch_size
        self.trainloader, self.testloader = self.load_data()

    def load_data(self) -> tuple:
        # Transform to tensor and normalize
        transform = transforms.ToTensor()

        download_data = True
        if os.path.exists(self.data_source):
            download_data = False
            print("MNIST dataset already exists. Skipping download.")

        trainset = datasets.MNIST(root='./data', train=True, download=download_data, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=download_data, transform=transform)

        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = DataLoader(testset, batch_size=64, shuffle=False)

        return trainloader, testloader