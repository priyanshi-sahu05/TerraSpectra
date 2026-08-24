import torch

from models.hybrid.cnn_vit import CNNViTHybrid


def test_hybrid_forward_pass():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    # Small mock hyperspectral input.
    x = torch.randn(
        2,
        1,
        8,
        8,
        8,
    )

    logits = model(x)

    print("Input shape:", x.shape)
    print("Hybrid output shape:", logits.shape)
    print("Logits:", logits)

    assert logits.shape == (2, 2)

def test_hybrid_probabilities():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    x = torch.randn(
        2,
        1,
        8,
        8,
        8,
    )

    logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    print("Probabilities:")
    print(probabilities)

    print("Predictions:")
    print(predictions)

    assert probabilities.shape == (2, 2)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )

    assert predictions.shape == (2,)