import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Load spectral patches
# ==========================================

with h5py.File("spectral_patches.h5", "r") as f:
    patches = f["patches"][:]
    labels = f["labels"][:]

print("Dataset shape:", patches.shape)
print("Labels shape:", labels.shape)


# ==========================================
# 2. Convert to PyTorch
# ==========================================

X = torch.tensor(
    patches.transpose(0, 3, 1, 2),
    dtype=torch.float32
)

y = torch.tensor(labels, dtype=torch.long)

print("Tensor shape:", X.shape)


# ==========================================
# 3. Train / Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. Lightweight ViT Baseline
# ==========================================

class SmallViT(nn.Module):

    def __init__(self, num_classes=2):
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

        self.classifier = nn.Linear(
            32,
            num_classes
        )

    def forward(self, x):

        # (B, 20, 5, 5)
        x = self.embedding(x)

        # (B, 32, 5, 5)
        x = x.flatten(2)

        # (B, 25, 32)
        x = x.transpose(1, 2)

        batch_size = x.size(0)

        cls = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat([cls, x], dim=1)

        x = x + self.position[:, :x.size(1)]

        x = self.transformer(x)

        x = x[:, 0]

        return self.classifier(x)


# ==========================================
# 5. Data loaders
# ==========================================

batch_size = 16

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ==========================================
# 6. Create ViT model
# ==========================================

model = SmallViT(num_classes=2)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 7. Train ViT
# ==========================================

epochs = 2

model.train()

for epoch in range(epochs):

    total_loss = 0
    correct = 0
    total = 0

    for batch_no, (batch_X, batch_y) in enumerate(train_loader):

        optimizer.zero_grad()

        output = model(batch_X)

        loss = criterion(output, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            output,
            dim=1
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

        if (batch_no + 1) % 500 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Batch {batch_no + 1}/{len(train_loader)}"
            )

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Loss: {total_loss / len(train_loader):.4f} - "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ==========================================
# 8. Evaluation
# ==========================================

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        output = model(batch_X)

        predictions = torch.argmax(
            output,
            dim=1
        )

        all_predictions.extend(
            predictions.numpy()
        )

        all_labels.extend(
            batch_y.numpy()
        )


accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n--- ViT Baseline Results ---")
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)