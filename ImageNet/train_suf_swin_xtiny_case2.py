import torch
from util.torch_dist_sum import *
from data.imagenet import *
from util.meter import *
import time
from util.accuracy import accuracy
from network.suf_swin_xti_case2 import *
import torch.nn as nn
from util.warmup_lr import *
from data.augmentation import get_deit_aug
import math
import torch.nn.functional as F
from util.mixup import Mixup
from util.weight_decay import create_params
from functools import partial
import numpy as np
import argparse

epochs = 300
warm_up = 20

current_folder='.'

parser = argparse.ArgumentParser("mobilenetv2_OneShot")
parser.add_argument('--break_val', type=int, default=-1, help='total steps in test')
parser.add_argument('--break_epoch', type=int, default=-1, help='total epochs in test')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
parser.add_argument('--ckpt_path', type=str, default='suf_swin_xtiny_case2', help='path to training dataset')

parser.add_argument('--port', type=int, default=23456, help='master port')

args = parser.parse_args()

model_name=args.ckpt_path

torch.autograd.set_detect_anomaly(True)

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

def soft_plus(x):
    result=torch.log(1+torch.exp(x))
    return result

def calculate_mi(p,q):
    term1=-1*torch.mean(soft_plus(-1*p))-torch.mean(soft_plus(q))
    return term1

def calculate_loss_network_mine(p_og,q_og,p_ig,q_ig,beta):
    term3=beta*calculate_mi(p_og,q_og)
    term4=beta*calculate_mi(p_ig,q_ig)
    output=term3-term4


    return output,term3,term4

def calculate_loss_mine(p_og,q_og,p_ig,q_ig):
    term3=calculate_mi(p_og,q_og)
    term4=calculate_mi(p_ig,q_ig)
    output=-term3-term4
    return output,term3,term4

def train(train_loader, model, MINE_OG_list, MINE_IG_list, local_rank, rank, optimizer,optimizer_mine, criterion, epoch, base_lr,batch_size):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_mineog0 = AverageMeter('Loss_mineog0', ':.4e')
    loss_mineog1 = AverageMeter('Loss_mineog1', ':.4e')
    loss_mineog2 = AverageMeter('Loss_mineog2', ':.4e')
    loss_mineig0 = AverageMeter('Loss_mineig0', ':.4e')
    loss_mineig1 = AverageMeter('Loss_mineig1', ':.4e')
    loss_mineig2 = AverageMeter('Loss_mineig2', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (samples, targets) in enumerate(train_loader):
        if i==args.break_val:
            break
        data_time.update(time.time() - end)

        if samples.shape[0]!=batch_size:
            break

        batch_x0 = samples.cuda(local_rank, non_blocking=True)
        batch_y0 = targets.cuda(local_rank, non_blocking=True)
        batch_x0, batch_y0 = mixup_fn(batch_x0, batch_y0)

        alpha_=1
        beta1=8/1000
        beta2=8/1000
        beta3=8/1000
        w1=1
        w2=1
        w3=1

        optimizer.zero_grad()

        # compute output
        prediction,x_0,x_00,x_01,x_1,x_10,x_11,x_2,x_20,x_21 = model(batch_x0)
        ord_list=np.arange(prediction.shape[0])
        np.random.shuffle(ord_list)

        prediction1=prediction.clone().detach()[ord_list]
        x1_0=x_0.clone().detach()[ord_list]
        x1_00=x_00.clone().detach()[ord_list]
        x1_01=x_01.clone().detach()[ord_list]
        x1_1=x_1.clone().detach()[ord_list]
        x1_10=x_10.clone().detach()[ord_list]
        x1_11=x_11.clone().detach()[ord_list]
        x1_2=x_2.clone().detach()[ord_list]
        x1_20=x_20.clone().detach()[ord_list]
        x1_21=x_21.clone().detach()[ord_list]

        m1_ogp=MINE_OG_list[0](x_00,x_01)
        m1_ogq=MINE_OG_list[0](x1_00,x_01)

        m1_igp=MINE_IG_list[0](x_0,x_01)
        m1_igq=MINE_IG_list[0](x1_0,x_01)

        m2_ogp=MINE_OG_list[1](x_10,x_11)
        m2_ogq=MINE_OG_list[1](x1_10,x_11)

        m2_igp=MINE_IG_list[1](x_1,x_11)
        m2_igq=MINE_IG_list[1](x1_1,x_11)

        m3_ogp=MINE_OG_list[2](x_20,x_21)
        m3_ogq=MINE_OG_list[2](x1_20,x_21)

        m3_igp=MINE_IG_list[2](x_2,x_21)
        m3_igq=MINE_IG_list[2](x1_2,x_21)

        loss_net1,term3_1,term4_1=calculate_loss_network_mine(
                        p_og=m1_ogp,
                        q_og=m1_ogq,
                        p_ig=m1_igp,
                        q_ig=m1_igq,
                        beta=beta1)
        loss_net2,term3_2,term4_2=calculate_loss_network_mine(
                        p_og=m2_ogp,
                        q_og=m2_ogq,
                        p_ig=m2_igp,
                        q_ig=m2_igq,
                        beta=beta2)
        loss_net3,term3_3,term4_3=calculate_loss_network_mine(
                        p_og=m3_ogp,
                        q_og=m3_ogq,
                        p_ig=m3_igp,
                        q_ig=m3_igq,
                        beta=beta3)

        loss_mineog0.update(term3_1.item(), batch_x0.size(0))
        loss_mineog1.update(term3_2.item(), batch_x0.size(0))
        loss_mineog2.update(term3_3.item(), batch_x0.size(0))
        loss_mineig0.update(term4_1.item(), batch_x0.size(0))
        loss_mineig1.update(term4_2.item(), batch_x0.size(0))
        loss_mineig2.update(term4_3.item(), batch_x0.size(0))

        loss = torch.sum(-batch_y0 * F.log_softmax(prediction, dim=-1), dim=-1).mean()
        losses.update(loss.item(), samples.size(0))

        loss_net=alpha_*loss+w1*loss_net1+w2*loss_net2+w3*loss_net3
        loss_net.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0)

        optimizer.step()
        adjust_learning_rate(optimizer, epoch, base_lr, int(i), int(len(train_loader)))

        optimizer_mine.zero_grad()
        m1_ogp=MINE_OG_list[0](x_00.detach(),x_01.detach())
        m1_ogq=MINE_OG_list[0](x1_00.detach(),x_01.detach())

        m1_igp=MINE_IG_list[0](x_0.detach(),x_01.detach())
        m1_igq=MINE_IG_list[0](x1_0.detach(),x_01.detach())

        m2_ogp=MINE_OG_list[1](x_10.detach(),x_11.detach())
        m2_ogq=MINE_OG_list[1](x1_10.detach(),x_11.detach())

        m2_igp=MINE_IG_list[1](x_1.detach(),x_11.detach())
        m2_igq=MINE_IG_list[1](x1_1.detach(),x_11.detach())

        m3_ogp=MINE_OG_list[2](x_20.detach(),x_21.detach())
        m3_ogq=MINE_OG_list[2](x1_20.detach(),x_21.detach())

        m3_igp=MINE_IG_list[2](x_2.detach(),x_21.detach())
        m3_igq=MINE_IG_list[2](x1_2.detach(),x_21.detach())

        loss_mine1,term3_1_,term4_1_=calculate_loss_mine(
                                    p_og=m1_ogp,
                                    q_og=m1_ogq,
                                    p_ig=m1_igp,
                                    q_ig=m1_igq)
        loss_mine2,term3_2_,term4_2_=calculate_loss_mine(
                                    p_og=m2_ogp,
                                    q_og=m2_ogq,
                                    p_ig=m2_igp,
                                    q_ig=m2_igq)
        loss_mine3,term3_3_,term4_3_=calculate_loss_mine(
                                    p_og=m3_ogp,
                                    q_og=m3_ogq,
                                    p_ig=m3_igp,
                                    q_ig=m3_igq)

        loss_mine=loss_mine1+loss_mine2+loss_mine3

        loss_mine.backward()



        optimizer_mine.step()
        adjust_learning_rate(optimizer_mine, epoch, base_lr, int(i), int(len(train_loader)))




        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0 and rank == 0:
            progress.display(i)

    return losses.avg, loss_mineog0.avg, loss_mineog1.avg, loss_mineog2.avg, loss_mineig0.avg, loss_mineig1.avg, loss_mineig2.avg


@torch.no_grad()
def test(test_loader, model, local_rank, rank, validate_loss_fn):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    val_loss_m = AverageMeter('Val_loss', ':.4e')

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
        output,x_0,x_00,x_01,x_1,x_10,x_11,x_2,x_20,x_21= model(img)
        val_loss = validate_loss_fn(output, target)
        val_loss_m.update(val_loss.item(), img.size(0))

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

    return top1_acc, top5_acc, val_loss_m.avg




def main():
    from torch.nn.parallel import DistributedDataParallel
    from util.dist_init import dist_init

    rank, local_rank, world_size = dist_init()
    batch_size = args.batch_size // world_size # single gpu
    num_workers = 4
    base_lr = 0.001
    weight_decay = 0.05

    model = SwinTransformer(img_size=224,
                                embed_dim=66,
                                depths=[ 2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
                                window_size=7,
                                drop_path_rate=0.2)
    model = model.cuda()

    MINE_OG_list=[]
    MINE_OG=mine_og0()
    MINE_OG=MINE_OG.cuda()
    MINE_OG_list.append(MINE_OG)
    MINE_OG=mine_og1()
    MINE_OG=MINE_OG.cuda()
    MINE_OG_list.append(MINE_OG)
    MINE_OG=mine_og2()
    MINE_OG=MINE_OG.cuda()
    MINE_OG_list.append(MINE_OG)


    MINE_IG_list=[]
    MINE_IG=mine_ig0()
    MINE_IG=MINE_IG.cuda()
    MINE_IG_list.append(MINE_IG)
    MINE_IG=mine_ig1()
    MINE_IG=MINE_IG.cuda()
    MINE_IG_list.append(MINE_IG)
    MINE_IG=mine_ig2()
    MINE_IG=MINE_IG.cuda()
    MINE_IG_list.append(MINE_IG)

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    params = create_params(model, weight_decay = weight_decay)
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    MINE_param=[]
    for model_mineog in MINE_OG_list:
        for p in model_mineog.parameters():
            MINE_param.append(p)
    for model_mineig in MINE_IG_list:
        for p in model_mineig.parameters():
            MINE_param.append(p)
    optimizer_mine = torch.optim.AdamW(MINE_param, lr=base_lr, weight_decay=weight_decay)

    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    for i in range(len(MINE_OG_list)):
        MINE_OG_list[i]=DistributedDataParallel(MINE_OG_list[i], device_ids=[local_rank],find_unused_parameters=True)
        MINE_IG_list[i]=DistributedDataParallel(MINE_IG_list[i], device_ids=[local_rank],find_unused_parameters=True)

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

    checkpoint_path = 'checkpoints/'+model_name+'.pth'
    print('checkpoint_path:', checkpoint_path)
    save_path = 'checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(checkpoint_path):
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        MINE_OG_list[0].load_state_dict(checkpoint['mine_og0'])
        MINE_OG_list[1].load_state_dict(checkpoint['mine_og1'])
        MINE_OG_list[2].load_state_dict(checkpoint['mine_og2'])
        MINE_IG_list[0].load_state_dict(checkpoint['mine_ig0'])
        MINE_IG_list[1].load_state_dict(checkpoint['mine_ig1'])
        MINE_IG_list[2].load_state_dict(checkpoint['mine_ig2'])
        optimizer_mine.load_state_dict(checkpoint['optimizer_mine'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    best_top1, best_top5 = 0, 0
    for epoch in range(start_epoch, epochs):
        if epoch==args.break_epoch:
            break
        train_sampler.set_epoch(epoch)
        losses_avg, loss_mineog0_avg, loss_mineog1_avg, loss_mineog2_avg, loss_mineig0_avg, loss_mineig1_avg, loss_mineig2_avg=train(train_loader, model, MINE_OG_list,MINE_IG_list, local_rank, rank, optimizer, optimizer_mine, criterion, epoch, base_lr , batch_size)
        with open(current_folder+'/'+model_name+'_stdout'+".txt", "a") as std_out:
            std_out.write(' epoch: '+str(epoch)+' train loss: '+str(losses_avg)+
                            ' og0: '+str(loss_mineog0_avg)+' ig0: '+str(loss_mineig0_avg)+
                            ' og1: '+str(loss_mineog1_avg)+' ig1: '+str(loss_mineig1_avg)+
                            ' og2: '+str(loss_mineog2_avg)+' ig2: '+str(loss_mineig2_avg)+'\n')
            std_out.close()
        top1, top5, val_loss = test(test_loader, model, local_rank, rank, validate_loss_fn)
        best_top1 = max(best_top1, top1)
        best_top5 = max(best_top5, top5)

        with open(current_folder+'/'+model_name+'_stdout'+".txt", "a") as std_out:
            std_out.write('val loss: '+str(val_loss)+
                            ' top1: '+str(top1)+' top5: '+str(top5)+
                            ' best_top1: '+str(best_top1)+' best_top5: '+str(best_top5)+'\n')
            std_out.close()

        if rank == 0:
            print('Epoch:{} * Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} Best_Acc@5 {:.3f}'.format(epoch, top1, top5, best_top1, best_top5))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mine_og0': MINE_OG_list[0].state_dict(),
                'mine_og1': MINE_OG_list[1].state_dict(),
                'mine_og2': MINE_OG_list[2].state_dict(),
                'mine_ig0': MINE_IG_list[0].state_dict(),
                'mine_ig1': MINE_IG_list[1].state_dict(),
                'mine_ig2': MINE_IG_list[2].state_dict(),
                'optimizer_mine': optimizer_mine.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)

if __name__ == "__main__":
    std_out=open(current_folder+'/'+model_name+'_stdout'+'.txt','w+')
    std_out.close()
    main()


