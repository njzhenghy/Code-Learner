from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import data_transforms
import sys


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, random_transform):
        self.random_transform = random_transform

    def __call__(self, x):
        q = self.random_transform(x)
        k = self.random_transform(x)
        return [q, k]


def data_loader(args):
    if args.dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = data_transforms(args)
        train_dataset = datasets.CIFAR10(root=args.data_root, train=True, 
                                         transform=TwoCropsTransform(transforms.Compose(train_transform)), download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        valid_dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=valid_transform, download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'mnist':
        num_classes = 10
        train_transform, valid_transform = data_transforms(args)
        train_dataset = datasets.MNIST(root=args.data_root, train=True, 
                                       transform=TwoCropsTransform(transforms.Compose(train_transform)), download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        valid_dataset = datasets.MNIST(root=args.data_root, train=False, transform=valid_transform, download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'tiny-imagenet':
        pass
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_transform, valid_transform = data_transforms(args)
        train_dataset = datasets.CIFAR100(root=args.data_root, train=True, transform=
                                          TwoCropsTransform(transforms.Compose(train_transform)), download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

        valid_dataset = datasets.CIFAR100(root=args.data_root, train=False, transform=valid_transform, download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print('dataset error')
        sys.exit(1)

    return num_classes, train_loader, valid_loader