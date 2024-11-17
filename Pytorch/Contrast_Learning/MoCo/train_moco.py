#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import logging
import torch, sys
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from model import ViT_encoder, MoCo
from utils import *
from data_loader import data_loader


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("-j", "--workers", default=32, type=int, metavar="N", help="number of data loading workers (default: 32)")
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", "--learning-rate", default=0.03, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int, help="learning rate schedule (when to drop lr by 10x)")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver")
parser.add_argument("--wd", "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-p", "--report-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("--seed", default=123, type=int, help="seed for initializing training.")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument("--moco-dim", default=1024, type=int, help="feature dimension (default: 128)")
parser.add_argument("--moco-k", default=65536, type=int, help="queue size; number of negative keys (default: 65536)")
parser.add_argument("--moco-m", default=0.999, type=float, help="moco momentum of updating key encoder (default: 0.999)",)
parser.add_argument("--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)")
# log storage
parser.add_argument('--img_size', type=int, default=32, help='image size')
parser.add_argument('--device', type=str, default='cuda', help='exp device')
parser.add_argument('--dataset', type=str, default='cifar10', help='Name of dataset')
parser.add_argument('--data_root', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--save', type=str, default='EXP', help='Experiment name')
# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument("--aug-plus", action="store_true", help="use moco v2 data augmentation")
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")
args = parser.parse_args()

log_save(args=args)
seed_torch(args.seed)   
if torch.cuda.is_available():
    device = torch.device(f"{args.device}:{args.gpu}")
elif torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    sys.exit(2)

encoder = ViT_encoder(image_size=args.img_size,
        patch_size=8,
        dim=args.moco_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1)

encoder.to(device)

# for name, param in encoder.named_parameters():
#     print(f"{name}: requires_grad = {param.requires_grad}")

model = MoCo(
    encoder,
    args.moco_dim,
    args.moco_k,
    args.moco_m,
    args.moco_t,
)

model.to(device)

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)

    # optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )

args.num_classes, train_loader, _ = data_loader(args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    total_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_loss = 0.
    # switch to train mode
    model.train()

    for step, (images, _) in enumerate(train_loader):

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        lr = adjust_learning_rate(optimizer, epoch, args)
        optimizer.zero_grad()

        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        batch_loss += loss.item()
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_n = images[0].size(0)

        total_loss.update(batch_loss, batch_n)
        top1.update(acc1[0], batch_n)
        top5.update(acc5[0], batch_n)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if step % args.report_freq == 0:
            logging.info('MoCo_train step:%03d batch_loss:%e top1_avg:%f top5_avg:%f', step, total_loss.avg, top1.avg, top5.avg)   

    return top1.avg, lr

if __name__ == "__main__":
    lr = args.lr
    best_acc = -1.
    for epoch in range(args.epochs):
        logging.info('Epoch %d lr %e', epoch, lr)
        train_acc, lr_tmp = train(train_loader, model, criterion, optimizer, epoch, args) 
        lr = lr_tmp
        if train_acc > best_acc:
            best_acc = train_acc
            save(model, os.path.join(args.save, 'pre_weights.pt'))