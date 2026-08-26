import torch


def predict_sample(
    model,
    input_tensor: torch.Tensor,
) -> dict:
    """
    Generate prediction for one sample.

    Expected input shape:

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

    if input_tensor.shape[0] != 1:
        raise ValueError(
        "predict_sample expects "
        "exactly one sample. "
        f"Received batch size "
        f"{input_tensor.shape[0]}"
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        class_id = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        probability = probabilities[
            0,
            class_id,
        ].item()

    return {
        "class_id": class_id,
        "class_name": (
            "Healthy"
            if class_id == 0
            else "Chemically Stressed"
        ),
        "probability": probability,
        "risk_score": probability,
    }


def predict_tiles(
    model,
    tiles,
):
    """
    Run model inference on each raster tile.

    Tile shape:

        [C, D, H, W]

    Model input:

        [B, C, D, H, W]
    """

    model.eval()

    results = []

    with torch.no_grad():

        for index, item in enumerate(
            tiles
        ):

            tile = item["tile"]

            # Add batch dimension
            model_input = tile.unsqueeze(0)

            logits = model(
                model_input
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            class_id = torch.argmax(
                probabilities,
                dim=1,
            ).item()

            probability = probabilities[
                0,
                class_id,
            ].item()

            results.append(
    {
        "tile_id": index,
        "row": item["row"],
        "col": item["col"],
        "height": item["height"],
        "width": item["width"],
        "class_id": class_id,
        "class_name": (
            "Healthy"
            if class_id == 0
            else "Chemically Stressed"
        ),
        "probability": probability,
    }
)

    return results