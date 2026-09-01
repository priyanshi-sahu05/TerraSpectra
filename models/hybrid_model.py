import torch
import torch.nn as nn


# =========================
# 1. 3D-CNN Feature Extractor
# =========================

class CNN3DFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


# =========================
# 2. ViT Feature Extractor
# =========================

class ViTFeatures(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Conv2d(
            in_channels=20,
            out_channels=32,
            kernel_size=1
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, 32)
        )

        self.position = nn.Parameter(
            torch.zeros(1, 26, 32)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

    def forward(self, x):
        # Expected input:
        # (B, 1, 20, 5, 5)

        x = x.squeeze(1)

        # (B, 20, 5, 5)
        x = self.embedding(x)

        # (B, 32, 5, 5)
        x = x.flatten(2)

        # (B, 32, 25)
        x = x.transpose(1, 2)

        # (B, 25, 32)

        batch_size = x.size(0)

        cls = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat([cls, x], dim=1)

        x = x + self.position[:, :x.size(1)]

        x = self.transformer(x)

        # CLS token feature
        x = x[:, 0]

        return x


# =========================
# 3. Hybrid CNN + ViT Model
# =========================

class HybridCNNViT(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = CNN3DFeatures()
        self.vit = ViTFeatures()

        # CNN = 32 features
        # ViT = 32 features
        # Combined = 64 features

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        cnn_features = self.cnn(x)

        vit_features = self.vit(x)

        combined = torch.cat(
            [cnn_features, vit_features],
            dim=1
        )

        output = self.classifier(combined)

        return output


# =========================
# 4. Test Hybrid Model
# =========================

if __name__ == "__main__":

    model = HybridCNNViT(num_classes=4)

    # Test input:
    # batch, channel, depth, height, width
    x = torch.randn(2, 1, 20, 5, 5)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Hybrid CNN + ViT forward pass successful!")

    # ==========================================
# 5. TRAIN HYBRID CNN + ViT MODEL
# ==========================================

import h5py
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Load spectral patches
with h5py.File("spectral_patches.h5", "r") as f:
    patches = f["patches"][:]
    labels = f["labels"][:]

print("Dataset shape:", patches.shape)
print("Labels shape:", labels.shape)

# Convert to PyTorch format
# Expected: [samples, 1, spectral_bands, height, width]
X = torch.tensor(
    patches.transpose(0, 3, 1, 2)[:, np.newaxis, :, :, :],
    dtype=torch.float32
)

y = torch.tensor(labels, dtype=torch.long)

print("Tensor shape:", X.shape)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

# Create Hybrid CNN + ViT model
model = HybridCNNViT(num_classes=4)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        output = model(batch_x)

        loss = criterion(output, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(output, dim=1)

        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy * 100:.2f}%"
    )

print("Hybrid CNN + ViT training completed!")

# ==============================
# 6. EVALUATION
# ==============================

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        output = model(batch_x)

        predictions = torch.argmax(output, dim=1)

        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total

print("\n--- Hybrid CNN + ViT Results ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")