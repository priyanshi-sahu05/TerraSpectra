import torch.nn as nn


class MockModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            1 * 8 * 8 * 8,
            2,
        )

    def forward(self, x):

        x = self.flatten(x)

        return self.classifier(x)