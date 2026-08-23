import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from models.cnn3d import Basic3DCNN


# -----------------------------
# 1. Load spectral patches
# -----------------------------
with h5py.File("spectral_patches.h5", "r") as f:
    patches = f["patches"][:]
    labels = f["labels"][:]

print("Dataset shape:", patches.shape)
print("Labels shape:", labels.shape)


# -----------------------------
# 2. Convert to PyTorch format
# (N, H, W, Bands)
# -> (N, 1, Bands, H, W)
# -----------------------------
X = torch.tensor(
    patches.transpose(0, 3, 1, 2),
    dtype=torch.float32
).unsqueeze(1)

y = torch.tensor(labels, dtype=torch.long)

print("Tensor shape:", X.shape)


# -----------------------------
# 3. Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# 4. Create CNN model
# -----------------------------
model = Basic3DCNN(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# 5. Initial CNN training
# -----------------------------
model.train()

epochs = 2

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X_train)

    loss = criterion(output, y_train)

    loss.backward()
    optimizer.step()

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Loss: {loss.item():.4f}"
    )


# -----------------------------
# 6. CNN Evaluation
# -----------------------------
model.eval()

with torch.no_grad():
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, dim=1)

    test_loss = criterion(predictions, y_test).item()


# -----------------------------
# 7. Accuracy
# -----------------------------
accuracy = accuracy_score(
    y_test.numpy(),
    predicted_labels.numpy()
)


# -----------------------------
# 8. Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    y_test.numpy(),
    predicted_labels.numpy()
)


print("\n--- CNN Evaluation Results ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy (%): {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)