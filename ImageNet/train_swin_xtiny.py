import torch
from util.torch_dist_sum import *
from data.imagenet import *
from util.meter import *
import time
from util.accuracy import accuracy
from network.swin_xtiny import *
import torch.nn as nn
from util.warmup_lr import *
from data.augmentation import get_deit_aug
import math
import torch.nn.functional as F
from util.mixup import Mixup
from util.weight_decay import create_params
from functools import partial
import argparse

epochs = 300
warm_up = 20

parser = argparse.ArgumentParser("mobilenetv2_OneShot")
parser.add_argument('--break_val', type=int, default=-1, help='total steps in test')
parser.add_argument('--break_epoch', type=int, default=-1, help='total epochs in test')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('--ckpt_path', type=str, default='swin_xtiny', help='path to training dataset')

parser.add_argument('--port', type=int, default=23456, help='master port')


args = parser.parse_args()

def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=1000)

def train(train_loader, model, local_rank, rank, optimizer, criterion, epoch, base_lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (samples, targets) in enumerate(train_loader):
        if i==args.break_val:
            break
        adjust_learning_rate(optimizer, epoch, base_lr, i, len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)

        samples = samples.cuda(local_rank, non_blocking=True)
        targets = targets.cuda(local_rank, non_blocking=True)

        samples, targets = mixup_fn(samples, targets)

        # compute output
        output = model(samples)
        loss = torch.sum(-targets * F.log_softmax(output, dim=-1), dim=-1).mean()
        losses.update(loss.item(), samples.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0 and rank == 0:
            progress.display(i)




@torch.no_grad()
def test(test_loader, model, local_rank, rank):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.eval()

    end = time.time()
    for i, (img, target) in enumerate(test_loader):
        if i==args.break_val:
            break
        # measure data loading time
        data_time.update(time.time() - end)


        img = img.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(img)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], img.size(0))
        top5.update(acc5[0], img.size(0))

        # compute gradient and do SGD step

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())

    return top1_acc, top5_acc




def main():
    from torch.nn.parallel import DistributedDataParallel
    from util.dist_init import dist_init

    rank, local_rank, world_size = dist_init()
    batch_size = args.batch_size // world_size # single gpu
    num_workers = 8
    base_lr = 0.001
    weight_decay = 0.05
    model = SwinTransformer(img_size=224,
                                embed_dim=66,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=7,
                                drop_path_rate=0.2)
    model = model.cuda()
    params = create_params(model, weight_decay = weight_decay)
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    torch.backends.cudnn.benchmark = True
    train_aug=get_deit_aug(res=224)
    print('please replace the following Imagenet training dataloader with your version, using the augmentation above.')
    train_dataset = IMAGENET_TRAIN_LOADER_PLACEHOLDER(train_aug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    default_res=224
    eval_res = 256 if default_res == 224 else 384
    test_aug = transforms.Compose([
        transforms.Resize(eval_res),
        transforms.CenterCrop(default_res),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    print('please replace the following Imagenet testing dataloader with your version, using the augmentation above.')
    test_dataset = IMAGENET_TEST_LOADER_PLACEHOLDER(test_aug)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler)

    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    checkpoint_path = 'checkpoints/'+args.ckpt_path+'.pth'
    print('checkpoint_path:', checkpoint_path)
    save_path = 'checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    best_top1, best_top5 = 0, 0
    for epoch in range(start_epoch, epochs):
        if epoch==args.break_epoch:
            break
        train_sampler.set_epoch(epoch)
        train(train_loader, model, local_rank, rank, optimizer, criterion, epoch, base_lr)
        top1, top5 = test(test_loader, model, local_rank, rank)
        best_top1 = max(best_top1, top1)
        best_top5 = max(best_top5, top5)

        if rank == 0:
            print('Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, top1, top5, best_top1, best_top5))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)

if __name__ == "__main__":
    main()


