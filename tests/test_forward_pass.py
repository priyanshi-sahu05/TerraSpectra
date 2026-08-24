import pytest
import torch

from models.hybrid import CNNViTHybrid


def create_model():

    return CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )


def create_input(batch_size=1):

    return torch.randn(
        batch_size,
        1,
        8,
        8,
        8,
    )


def test_single_sample_forward_pass():

    model = create_model()

    x = create_input(
        batch_size=1
    )

    logits = model(x)

    print("\nSingle sample:")
    print("Input:", x.shape)
    print("Output:", logits.shape)

    assert logits.shape == (
        1,
        2,
    )


def test_batch_forward_pass():

    model = create_model()

    x = create_input(
        batch_size=4
    )

    logits = model(x)

    print("\nBatch:")
    print("Input:", x.shape)
    print("Output:", logits.shape)

    assert logits.shape == (
        4,
        2,
    )


def test_forward_output_is_finite():

    model = create_model()

    x = create_input()

    logits = model(x)

    assert torch.isfinite(
        logits
    ).all()


def test_forward_output_contains_no_nan():

    model = create_model()

    x = create_input()

    logits = model(x)

    assert not torch.isnan(
        logits
    ).any()


def test_forward_output_contains_no_inf():

    model = create_model()

    x = create_input()

    logits = model(x)

    assert not torch.isinf(
        logits
    ).any()

def test_different_batch_sizes():

    model = create_model()

    for batch_size in [1, 2, 4, 8]:

        x = create_input(
            batch_size=batch_size
        )

        logits = model(x)

        print(
            f"\nBatch size {batch_size}:",
            logits.shape,
        )

        assert logits.shape == (
            batch_size,
            2,
        )

        assert torch.isfinite(
            logits
        ).all()

def test_evaluation_forward_pass():

    model = create_model()

    model.eval()

    x = create_input()

    with torch.no_grad():

        logits = model(x)

    assert model.training is False

    assert logits.shape == (
        1,
        2,
    )

    assert torch.isfinite(
        logits
    ).all()

def test_softmax_probability_output():

    model = create_model()

    model.eval()

    x = create_input()

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    print(
        "\nLogits:",
        logits,
    )

    print(
        "Probabilities:",
        probabilities,
    )

    assert probabilities.shape == (
        1,
        2,
    )

    assert torch.all(
        probabilities >= 0
    )

    assert torch.all(
        probabilities <= 1
    )

    total = probabilities.sum(
        dim=1
    )

    assert torch.allclose(
        total,
        torch.ones_like(total),
        atol=1e-6,
    )

def test_gradient_flow():

    model = create_model()

    model.train()

    x = create_input()

    logits = model(x)

    target = torch.tensor(
        [1],
        dtype=torch.long,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    loss = loss_function(
        logits,
        target,
    )

    loss.backward()

    assert loss.item() >= 0

    gradient_found = False

    for parameter in model.parameters():

        if (
            parameter.requires_grad
            and parameter.grad is not None
        ):

            gradient_found = True

            assert torch.isfinite(
                parameter.grad
            ).all()

    assert gradient_found

def test_large_input_values():

    model = create_model()

    x = torch.randn(
        1,
        1,
        8,
        8,
        8,
    ) * 10

    logits = model(x)

    assert torch.isfinite(
        logits
    ).all()

def test_invalid_input_dimensions():

    model = create_model()

    invalid_input = torch.randn(
        1,
        8,
        8,
        8,
    )

    with pytest.raises(ValueError):

        model(invalid_input)