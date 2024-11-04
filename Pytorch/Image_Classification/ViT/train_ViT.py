import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from model import ViT
from dataloader import train_loader_mnist, val_loader_mnist

arg = argparse.ArgumentParser()
arg.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
arg.add_argument('--lr', type=float, default=0.01, help='Learning rate')
arg.add_argument('-batch_size', type=int, default=16, help='Batch size')
arg.add_argument('--seed', type=int, default=32, help='Random seed')
arg.add_argument('--device', type=str, default='cuda', help='Device used for training & validation')
arg.add_argument('--gpu', type=int, default=0, help='GPU ID')
arg.add_argument('--img_size', type=int, default=28, help='Image size')
arg.add_argument('--in_channels', type=int, default=3, help='Image channels')
arg.add_argument('--num_classes', type=int, default=10, help='Number of classes')
arg.add_argument('--patch_size', type=int, default=8, help='Patch size')
arg.add_argument('--depth', type=int, default=6, help='Depth')
arg.add_argument('--heads', type=int, default=4, help='Number of heads')
arg.add_argument('--dim', type=int, default=8, help='ViT Dimension')
arg.add_argument('--mlp_dim', type=int, default=16, help='MLP Dimension')
args = arg.parse_args()

model = ViT(
    image_size=args.img_size,
    patch_size=args.patch_size,
    num_classes=args.num_classes,
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    dropout=0.1,
    emb_dropout=0.1
)

if torch.cuda.is_available():
    device = torch.device(f"{args.device}:{args.gpu}")
elif torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


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