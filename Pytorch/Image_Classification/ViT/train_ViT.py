import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from model import ViT
from data_loader import _data_loader
import sys, os, time, logging
import torch.backends.cudnn as cudnn
import numpy as np
from utils import AvgrageMeter, _accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--seed', type=int, default=32, help='Random seed')
parser.add_argument('--device', type=str, default='cuda', help='Device used for training & validation')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--img_size', type=int, default=28, help='Image size')
parser.add_argument('--in_channels', type=int, default=3, help='Image channels')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--patch_size', type=int, default=8, help='Patch size')
parser.add_argument('--depth', type=int, default=6, help='Depth')
parser.add_argument('--heads', type=int, default=4, help='Number of heads')
parser.add_argument('--dim', type=int, default=8, help='ViT Dimension')
parser.add_argument('--mlp_dim', type=int, default=16, help='MLP Dimension')
parser.add_argument('--dataset', type=str, default='cifar10', help='Name of dataset')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
args = parser.parse_args()

args.save = 'eval_online_{}-{}-{}'.format(args.data_name, args.save, time.strftime("%Y%m%d-%H"))
if not os.path.exists(args.save):
    os.makedirs(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if torch.cuda.is_available():
    device = torch.device(f"{args.device}:{args.gpu}")
elif torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    sys.exit(2)

np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
cudnn.benchmark = True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)


train_loader, valid_loader = _data_loader(args)
model = ViT(
    image_size=args.img_size,
    patch_size=args.patch_size,
    num_classes=args.num_classes,
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    dropout=args.dropout,
    emb_dropout=args.dropout,)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
criterion = nn.CrossEntropyLoss()
best_acc = 0.
model.to(device)


def train(loader):
    model.train()
    batch_loss = 0
    total_loss = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad(model.parameters, args.grad_clip)
        optimizer.step()
        
        prec1, prec5 = _accuracy(outputs, labels, topk=(1, 5), batch_size=args.batch_size)
        total_loss.update(batch_loss, n=args.batch_size)
        top1.update(prec1, n=args.batch_size)
        top5.update(prec5, n=args.batch_size)

    return step, total_loss.avg, top1.avg, top5.avg




if __name__ == "__main__": 
    for epoch in range(args.epochs):
        logging.info('Epoch %d lr %e', epoch, scheduler.get_lr()[0])
        step, total_loss, top1, top5 = train(train_loader) 
        if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f ', step, total_loss, top1, top5)

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_images, _ in valid_loader:
                    val_images = val_images.to(device)
                    val_outputs = model(val_images)
                    val_loss += criterion(val_outputs, val_images).item()

            avg_val_loss = val_loss / len(valid_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")