import torch
import torch.nn as nn


class CNNToViTAdapter(nn.Module):
    """
    Converts CNN feature channels into the feature dimension
    expected by the Vision Transformer.

    Current mock CNN:
        [B, 16, D, H, W]

    Adapter output:
        [B, 16, D, H, W]

    The channel dimensions are configurable so that the adapter
    can later be changed easily when Member 2 provides the
    real CNN.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.projection = nn.Conv3d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "Expected CNN features with shape "
                "[B, C, D, H, W], "
                f"but received {tuple(x.shape)}"
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} CNN channels, "
                f"but received {x.shape[1]}"
            )

        return self.projection(x)