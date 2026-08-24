import torch

from models.vit.vit import VisionTransformer


def test_vit_forward_pass():

    model = VisionTransformer(
        input_channels=16,
        num_classes=2,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
    )

    # Controlled mock CNN feature tensor.
    x = torch.randn(
        2,
        16,
        16,
        16,
        16,
    )

    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    assert logits.shape == (2, 2)

def test_vit_probability_output():

    model = VisionTransformer(
        input_channels=16,
        num_classes=2,
    )

    x = torch.randn(
        2,
        16,
        16,
        16,
        16,
    )

    logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    print("Probabilities:", probabilities)
    print(
        "Probability sums:",
        probabilities.sum(dim=1),
    )

    assert probabilities.shape == (2, 2)

    # Each sample's class probabilities should sum to 1.
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )

def test_vit_prediction():

    model = VisionTransformer(
        input_channels=16,
        num_classes=2,
    )

    x = torch.randn(
        2,
        16,
        16,
        16,
        16,
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

    print("Predicted classes:", predictions)

    assert predictions.shape == (2,)

    for prediction in predictions:
        assert prediction.item() in [0, 1]

from models.vit import CLASS_LABELS


def test_class_labels():

    assert CLASS_LABELS[0] == "Healthy"
    assert CLASS_LABELS[1] == "Chemically Stressed"

    print("Class labels:", CLASS_LABELS)
