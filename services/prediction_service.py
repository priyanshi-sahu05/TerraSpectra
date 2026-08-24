import torch

from inference.predict import predict_sample
from models.hybrid import CNNViTHybrid


class PredictionService:
    """
    Handles model creation and prediction.

    This service will later be called by FastAPI.
    """

    def __init__(self):

        self.model = CNNViTHybrid(
            cnn_channels=16,
            vit_channels=16,
            num_classes=2,
        )

        self.model.eval()

    def predict(
        self,
        input_tensor: torch.Tensor,
    ) -> dict:
        """
        Generate prediction for one sample.
        """

        return predict_sample(
            self.model,
            input_tensor,
        )

    def predict_batch(
        self,
        input_tensor: torch.Tensor,
    ) -> list:
        """
        Generate predictions for multiple samples.

        Expected shape:

        [B, C, D, H, W]
        """

        if not isinstance(
            input_tensor,
            torch.Tensor,
        ):
            raise TypeError(
                "input_tensor must be a torch.Tensor"
            )

        if input_tensor.ndim != 5:
            raise ValueError(
                "Expected input shape "
                "[B, C, D, H, W]. "
                f"Received "
                f"{tuple(input_tensor.shape)}"
            )

        results = []

        for index in range(
            input_tensor.shape[0]
        ):

            sample = input_tensor[
                index:index + 1
            ]

            result = self.predict(
                sample
            )

            result["sample_index"] = index

            results.append(result)

        return results