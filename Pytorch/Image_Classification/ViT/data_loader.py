import ssl
import sys
from torchvision import datasets
from torch.utils.data import DataLoader
from utils import _data_transforms

ssl._create_default_https_context = ssl._create_unverified_context
def _data_loader(args):
    if args.dataset == 'cifar10':
        train_transform, valid_transform = _data_transforms(args)
        train_dataset = datasets.CIFAR10(root=args.data_root, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        valid_dataset = datasets.CIFAR10(root=args.data_root, train=False, transform=valid_transform, download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'mnist':
        train_transform, valid_transform = _data_transforms(args)
        train_dataset = datasets.MNIST(root=args.data_root, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        valid_dataset = datasets.MNIST(root=args.data_root, train=False, transform=valid_transform, download=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    elif args.dataset == 'tiny-imagenet':
        pass

    else:
        print('dataset error')
        sys.exit(1)
    
    return train_loader, valid_loader