import torch
import torch.nn as nn

from models.mock_cnn import MockCNN3D
from models.vit.vit import VisionTransformer
from models.hybrid.adapter import CNNToViTAdapter


class CNNViTHybrid(nn.Module):
    """
    CNN + Vision Transformer hybrid model.

    Current development version uses MockCNN3D because
    Member 2's real CNN is not available yet.

    Pipeline:

        Input
          ↓
        CNN
          ↓
        CNN features
          ↓
        CNN → ViT adapter
          ↓
        Vision Transformer
          ↓
        Classification
    """

    def __init__(
        self,
        cnn_channels: int = 16,
        vit_channels: int = 16,
        num_classes: int = 2,
    ):
        super().__init__()

        self.cnn_channels = cnn_channels
        self.vit_channels = vit_channels
        self.num_classes = num_classes

        # Temporary CNN.
        self.cnn = MockCNN3D(
            num_classes=num_classes,
        )

        # Converts CNN feature channels into the
        # representation expected by the ViT.
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

    def extract_cnn_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract feature representation from the CNN.
        """

        if x.ndim != 5:
            raise ValueError(
                "Expected input shape [B, C, D, H, W], "
                f"but received {tuple(x.shape)}"
            )

        features = self.cnn.forward_features(x)

        if features.ndim != 5:
            raise ValueError(
                "CNN features must have shape "
                "[B, C, D, H, W]. "
                f"Received {tuple(features.shape)}"
            )

        return features

    def adapt_features(
        self,
        cnn_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert CNN features into the representation
        expected by the ViT.
        """

        if cnn_features.ndim != 5:
            raise ValueError(
                "Expected CNN features with shape "
                "[B, C, D, H, W]."
            )

        return self.adapter(cnn_features)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Complete CNN → Adapter → ViT forward pass.
        """

        cnn_features = self.extract_cnn_features(x)

        vit_features = self.adapt_features(
            cnn_features
        )

        logits = self.vit(vit_features)

        return logits