import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, hidden_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # PIPELINING

    def encode(self, x):
        return self.encoder(x)
    

class Autoencoder_PIPE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_PIPE, self).__init__()
        self.encoder_PIPE1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
        )
        self.encoder_PIPE2 = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(True),
        )
        self.decoder_PIPE1 = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(True),
        )
        self.decoder_PIPE2 = nn.Sequential(
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded_1 = self.encoder_PIPE1(x)
        encoded_2 = self.encoder_PIPE2(encoded_1)
        decoded_1 = self.decoder_PIPE1(encoded_2)
        decoded_2 = self.decoder_PIPE2(decoded_1)
        return decoded_2
    
    def forward_step0(self, x):
        encoded_1 = self.encoder_PIPE1(x)
        return encoded_1
    
    def forward_step1(self, x):
        encoded_2 = self.encoder_PIPE2(x)
        return encoded_2
    
    def forward_step2(self, x):
        decoded_1 = self.decoder_PIPE1(x)
        return decoded_1
    
    def forward_step3(self, x):
        decoded_2 = self.decoder_PIPE2(x)
        return decoded_2
    
    # PIPELINING

    def encode(self, x):
        return self.encoder(x)

# AUTOENCODER LAYERS WITH CLASSES


class Layer0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.layer(x)

class Layer1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.layer(x)

class Layer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.layer(x)

class Layer3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layer(x)  # output layer (no activation)


# Define the two stages
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(256, 784),
            nn.Sigmoid(),  # Or remove and use BCEWithLogitsLoss
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        return self.layer(x)