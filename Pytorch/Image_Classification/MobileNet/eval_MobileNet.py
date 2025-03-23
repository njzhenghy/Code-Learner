import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from Pytorch.Image_Classification.MobileNet.model_MobileNet import MobileNetV2
from data_loader import data_loader
import sys, os, time, logging
from utils import AverageMeter, accuracy, seed_torch, save, load

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--seed', type=int, default=32, help='Random seed')
parser.add_argument('--device', type=str, default='cuda', help='Device used for training & validation')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--img_size', type=int, default=28, help='Image size')
parser.add_argument('--in_channels', type=int, default=3, help='Image channels')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
parser.add_argument('--depth', type=int, default=12, help='Depth')
parser.add_argument('--heads', type=int, default=12, help='Number of heads')
parser.add_argument('--hidden_dim', type=int, default=768, help='ViT Hidden Dimension')
parser.add_argument('--mlp_dim', type=int, default=3072, help='MLP Dimension')
parser.add_argument('--dataset', type=str, default='cifar10', help='Name of dataset')
parser.add_argument('--cutout', action='store_true', default=False, help='Use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='Cutout length')
parser.add_argument('--save', type=str, default='EXP', help='Experiment name')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--grad_clip', type=float, default=5, help='Gradient clipping')
parser.add_argument('--report_freq', type=float, default=10, help='Report frequency')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay of SGD')  
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_root', type=str, default='.\__checkpoint__\weights.pt', help='weight file') 
args = parser.parse_args()

args.save = 'test_MobileNetv2_{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(args.save):
    os.makedirs(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'lr{}_wd{}_batch{}.txt'.format(args.lr, args.weight_decay, args.batch_size)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if torch.cuda.is_available():
    device = torch.device(f"{args.device}:{args.gpu}")
elif torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    sys.exit(2)


logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)
seed_torch(args.seed)

args.num_classes, _, test_loader = data_loader(args)

model = MobileNetV2()

load(model, args.weight_root)

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
criterion = nn.CrossEntropyLoss()
 

def infer(test_loader, model, optimizer, criterion):
    model.eval()
    total_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for step, (images, labels) in enumerate(test_loader):
        batch_loss = 0
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5), batch_size=args.batch_size)
        batch_n = images.size(0)
        total_loss.update(batch_loss, n=batch_n)
        top1.update(prec1, n=batch_n)
        top5.update(prec5, n=batch_n)

        if step % args.report_freq == 0:
            logging.info('eval step:%03d batch_loss:%e top1_avg:%f top5_avg:%f', step, total_loss.avg, top1.avg, top5.avg)    
    
    return top1.avg

if __name__ == "__main__": 
    test_acc = infer(test_loader, model, optimizer, criterion)
    logging.info('Test_acc %f', test_acc)
