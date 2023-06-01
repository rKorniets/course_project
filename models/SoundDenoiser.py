from torch import nn
class SoundDenoiser(nn.Module):
    def __init__(self, input_size=128):
        super(SoundDenoiser, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, input_size, kernel_size=17, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_size, 64, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_size, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(input_size, 1, kernel_size=17, stride=4, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded