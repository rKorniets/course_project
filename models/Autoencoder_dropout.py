from torch import nn

class Autoencoder_dropout(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder_dropout, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),  # Додано Dropout з ймовірністю 0.2
            nn.ConvTranspose1d(128, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded