import torch


def test_pytorch_installation():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    assert torch.__version__ is not None