import torch
import torch.nn as nn


class Basic3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = Basic3DCNN(num_classes=4)

    # Test input: batch, channels, depth, height, width
    x = torch.randn(2, 1, 8, 32, 32)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Forward pass successful!")