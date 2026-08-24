import torch
import torch.nn as nn


class MockCNN3D(nn.Module):
    """
    Temporary mock CNN used for Member 3 development.

    This is NOT Member 2's final 3D-CNN.

    Its purpose is to provide a controlled feature tensor so
    the CNN -> ViT pipeline can be developed and tested.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.classifier = nn.AdaptiveAvgPool3d(1)

        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return CNN feature representation.

        Input:
            [B, 1, D, H, W]

        Output:
            [B, 16, D, H, W]
        """

        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return classification logits.
        """

        features = self.forward_features(x)

        pooled = self.classifier(features)

        pooled = pooled.flatten(1)

        # Temporary classification layer.
        # This will eventually be replaced by the hybrid model.
        return pooled