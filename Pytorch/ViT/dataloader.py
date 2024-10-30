from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    # transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 如果需要
])

transform_cifar = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_imagenet = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 使用 ImageNet 预训练的标准化
])

# MNIST
train_dataset_mnist = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
train_loader_mnist = DataLoader(train_dataset_mnist, batch_size=1, shuffle=True)

# CIFAR-10
train_dataset_cifar = datasets.CIFAR10(root='./data', train=True, transform=transform_cifar, download=True)
train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=4, shuffle=True)

# # ImageNet
# train_dataset_imagenet = datasets.ImageFolder(root='path/to/imagenet/train', transform=transform_imagenet)
# train_loader_imagenet = DataLoader(train_dataset_imagenet, batch_size=4, shuffle=True)