import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):

    def __init__(
        self,
        input_channels: int = 16,
        num_classes: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pool_size = (4, 4, 4)

        self.feature_projection = nn.Linear(
            input_channels,
            embed_dim,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.position_embedding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

    def _create_position_embedding(
        self,
        num_tokens: int,
        device: torch.device,
    ):
        if (
            self.position_embedding is None
            or self.position_embedding.shape[1] != num_tokens
        ):
            self.position_embedding = nn.Parameter(
                torch.zeros(
                    1,
                    num_tokens,
                    self.embed_dim,
                    device=device,
                )
            )

            nn.init.trunc_normal_(
                self.position_embedding,
                std=0.02,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 5:
            raise ValueError(
                "Expected input with shape [B, C, D, H, W], "
                f"but received {tuple(x.shape)}"
            )

        batch_size, channels, depth, height, width = x.shape

        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, "
                f"but received {channels}"
            )

        x = F.adaptive_avg_pool3d(
            x,
            self.pool_size,
        )

        x = x.permute(
            0,
            2,
            3,
            4,
            1,
        )

        x = x.reshape(
            batch_size,
            -1,
            channels,
        )

        x = self.feature_projection(x)

        cls_token = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [cls_token, x],
            dim=1,
        )

        self._create_position_embedding(
            x.shape[1],
            x.device,
        )

        x = x + self.position_embedding

        x = self.transformer(x)

        cls_output = x[:, 0]

        cls_output = self.norm(
            cls_output
        )

        logits = self.classifier(
            cls_output
        )

        return logits