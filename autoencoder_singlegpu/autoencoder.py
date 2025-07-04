import torch
import torch.nn as nn

class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ManualLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ManualLinear(28*28, 256),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            ManualLinear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # PIPELINING

    def encode(self, x):
        return self.encoder(x)
    
