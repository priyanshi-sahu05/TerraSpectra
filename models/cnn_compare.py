import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Load dataset
with h5py.File("spectral_patches.h5", "r") as f:
    X = np.array(f["patches"], dtype=np.float32)
    y = np.array(f["labels"], dtype=np.int64)

print("Dataset:", X.shape)
print("Labels:", y.shape)

# Convert:
# (N, 5, 5, 20) -> (N, 1, 20, 5, 5)
X = torch.tensor(X).permute(0, 3, 1, 2).unsqueeze(1)
y = torch.tensor(y)

# Normalize
X = (X - X.mean()) / (X.std() + 1e-8)

dataset = TensorDataset(X, y)

# Same train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Basic 3D CNN
class Basic3DCNN(nn.Module):
    def __init__(self, num_classes=4):
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

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = Basic3DCNN(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):

    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        output = model(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(output, dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    train_accuracy = correct / total * 100

    print(
        f"Epoch [{epoch+1}/5] "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"Accuracy: {train_accuracy:.2f}%"
    )


# Testing
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output = model(batch_x)

        predictions = torch.argmax(output, dim=1)

        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

cnn_accuracy = correct / total * 100

print("\n--- CNN Results ---")
print(f"CNN Test Accuracy: {cnn_accuracy:.2f}%")
print("Comparison: CNN vs Hybrid CNN + ViT")