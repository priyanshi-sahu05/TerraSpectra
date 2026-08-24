import torch

from models.vit import CLASS_LABELS


def predict_sample(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> dict:
    """
    Run inference on one hyperspectral sample.

    Expected input:
        [1, C, D, H, W]

    Returns:
        Structured prediction dictionary.
    """

    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(
            "input_tensor must be a torch.Tensor"
        )

    if input_tensor.ndim != 5:
        raise ValueError(
            "Expected input shape [B, C, D, H, W], "
            f"but received {tuple(input_tensor.shape)}"
        )

    if input_tensor.shape[0] != 1:
        raise ValueError(
            "predict_sample expects exactly one sample. "
            f"Received batch size {input_tensor.shape[0]}."
        )

    model.eval()

    with torch.no_grad():

        logits = model(input_tensor)

        if logits.ndim != 2:
            raise ValueError(
                "Model output must have shape [B, num_classes]. "
                f"Received {tuple(logits.shape)}"
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_class_id = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        predicted_probability = probabilities[
            0,
            predicted_class_id,
        ].item()

        # Class 1 represents chemically stressed
        # in the current mock setup.
        risk_score = probabilities[
            0,
            1,
        ].item()

    class_name = CLASS_LABELS.get(
        predicted_class_id,
        "Unknown",
    )

    return {
        "class_id": predicted_class_id,
        "class_name": class_name,
        "probability": round(
            predicted_probability,
            4,
        ),
        "risk_score": round(
            risk_score,
            4,
        ),
    }