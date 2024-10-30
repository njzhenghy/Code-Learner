import torch
import torch.optim as optim
import torch.nn as nn
from model import ViT
from dataloader import train_loader_mnist, val_loader_mnist

model = ViT(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=8,
    depth=1,
    heads=8,
    mlp_dim=16,
    dropout=0.1,
    emb_dropout=0.1
)

device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training loop
def train(loader):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(loader.dataset)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


print("start training...")
# Example usage:
epochs = 20
for epoch in range(epochs):
    avg_loss, accuracy = train(train_loader_mnist)  # For MNIST
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_images, _ in val_loader_mnist:
                val_images = val_images.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_images).item()

        avg_val_loss = val_loss / len(val_loader_mnist)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")