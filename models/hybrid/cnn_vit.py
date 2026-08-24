import torch
import torch.nn as nn

from models.mock_cnn import MockCNN3D
from models.vit.vit import VisionTransformer
from models.hybrid.adapter import CNNToViTAdapter


class CNNViTHybrid(nn.Module):
    """
    Temporary CNN + ViT hybrid model.

    Current development uses the mock CNN because the real
    Member 2 CNN has not yet been implemented.
    """

    def __init__(
        self,
        cnn_channels: int = 16,
        vit_channels: int = 16,
        num_classes: int = 2,
    ):
        super().__init__()

        # Temporary CNN.
        self.cnn = MockCNN3D(
            num_classes=num_classes,
        )

        # CNN -> ViT feature adapter.
        self.adapter = CNNToViTAdapter(
            input_channels=cnn_channels,
            output_channels=vit_channels,
        )

        # Vision Transformer.
        self.vit = VisionTransformer(
            input_channels=vit_channels,
            num_classes=num_classes,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Step 1: extract CNN features.
        cnn_features = self.cnn.forward_features(x)

        # Step 2: adapt CNN features for ViT.
        vit_features = self.adapter(cnn_features)

        # Step 3: Transformer classification.
        logits = self.vit(vit_features)

        return logits