from .vit import VisionTransformer

CLASS_LABELS = {
    0: "Healthy",
    1: "Chemically Stressed",
}

__all__ = [
    "VisionTransformer",
    "CLASS_LABELS",
]