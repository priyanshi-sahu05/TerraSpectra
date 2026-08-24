import torch

from models.vit import CLASS_LABELS


def predict_sample(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> dict:
    """
    Run inference on a single mock hyperspectral sample.

    Parameters
    ----------
    model:
        Hybrid CNN + ViT model.

    input_tensor:
        Tensor with shape [1, 1, D, H, W].

    Returns
    -------
    dict:
        Structured prediction result.
    """

    # Validate tensor dimensions.
    if input_tensor.ndim != 5:
        raise ValueError(
            "Expected input shape [B, C, D, H, W], "
            f"but received {tuple(input_tensor.shape)}"
        )

    # This function currently handles one sample at a time.
    if input_tensor.shape[0] != 1:
        raise ValueError(
            "predict_sample expects exactly one sample. "
            f"Received batch size {input_tensor.shape[0]}."
        )

    # Switch model to evaluation mode.
    model.eval()

    # No gradients are required during inference.
    with torch.no_grad():

        # Run the hybrid CNN + ViT model.
        logits = model(input_tensor)

        # Convert logits into probabilities.
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        # Find the class with the highest probability.
        predicted_class_id = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        # Probability of the predicted class.
        predicted_probability = probabilities[
            0,
            predicted_class_id,
        ].item()

        # Class 1 represents "Chemically Stressed".
        # Therefore its probability is used as the risk score.
        risk_score = probabilities[
            0,
            1,
        ].item()

    # Convert class ID into a readable class name.
    class_name = CLASS_LABELS.get(
        predicted_class_id,
        "Unknown",
    )

    # Return structured prediction information.
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