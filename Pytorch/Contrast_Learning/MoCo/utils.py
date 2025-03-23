import torch
import math, os, logging, time, sys
import numpy as np
from torchvision import transforms
import random
from PIL import ImageFilter

def save_checkpoint(model, model_path):
    torch.save(model.state_dict(), model_path)


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val =val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def seed_torch(seed=21):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 的 benchmark 模式，以确保每次运行都有相同的结果
    torch.backends.cudnn.deterministic = True  # 启用 CuDNN 的确定性模式，进一步确保可重复性


def log_save(args):
    args.save = 'train_ViT_{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'lr{}_wd{}_batch{}.txt'.format(args.lr, args.weight_decay, args.batch_size)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def data_transforms(args):
    train_transform, valid_transform = None, None

    if args.dataset == 'mnist':
        MNIST_MEAN = (0.1307,)
        MNIST_STD = (0.3081,)
        normalize = transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD)
        img_size = 28

    if args.dataset == 'cifar10':
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        normalize = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        img_size = 32

    if args.dataset == 'cifar100':
        CIFAR100_MEAN = [0.50707516, 0.48654887, 0.44091784]
        CIFAR100_STD = [0.2673344 , 0.25643831, 0.27615047]
        normalize = transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        img_size = 32

    if args.dataset == 'tiny-imagenet':
        pass

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        train_transform = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        train_transform = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    
    valid_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), normalize])

    return train_transform, valid_transform

def save(model, model_path):
    torch.save(model.state_dict(), model_path)