import torch
from util.torch_dist_sum import *
from data.imagenet import *
from util.meter import *
import time
from util.accuracy import accuracy
import torch.nn as nn
from util.warmup_lr import *
from data.augmentation import get_deit_aug
import math
import torch.nn.functional as F
from util.mixup import Mixup
from util.weight_decay import create_params
from functools import partial
import numpy as np
import torchvision
from torchvision import transforms
import argparse

#import different architectures
from network.suf_vit_ti import suf_vit_ti


parser = argparse.ArgumentParser("mobilenetv2_OneShot")

#change when test
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--break_val', default=-1, type=int)

#change everytime
parser.add_argument('--model_name', default='suf_vit_ti',type=str,
                help='device to use for training / testing')
parser.add_argument('--checkpoint_path', default='insert_checkpoint_path_here',type=str,
                help='where to load model')

#no need to change
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--pin-mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

parser.add_argument('--port', type=int, default=23456, help='master port')

args = parser.parse_args()

epochs = 300
warm_up = 5

current_folder='.'
task_name='try_cifar'
checkpoint_path=args.checkpoint_path




def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def attack(model,input,label,norm_list,loss_func):
    model.train()
    input.requires_grad = True
    if args.model_name=='suf_vit_ti' or args.model_name=='improved4':
        prediction=model(input)
        prediction=prediction[0]
    else:
        prediction=model(input)
    label_np=label.cpu().detach().numpy()
    target_label=[]
    for i in range(len(label_np)):
        current_label=np.random.rand()*10
        while current_label==label_np[i]:
            current_label=np.random.rand()*10
        target_label.append(current_label)
    target_label=torch.from_numpy(np.asarray(target_label)).type(torch.int64).cuda()

    loss=loss_func(prediction,target_label)
    model.zero_grad()
    loss.backward()
    data_grad = input.grad.data

    model.eval()
    loss_list=[]
    acc_list=[]
    counter_list=[]
    for i in range(len(norm_list)):
        val_loss_sum=0
        val_acc_sum=0
        counter=0

        if norm_list[i]==0:
            perturbed_data=input
        else:
            perturbed_data = fgsm_attack(input, norm_list[i], data_grad)

        if args.model_name=='suf_vit_ti' or args.model_name=='improved4':
            prediction_v=model(perturbed_data)
            prediction_v=prediction_v[0]
        else:
            prediction_v=model(perturbed_data)
        loss_v = loss_func(prediction_v,label)
        loss_vt=loss_v.cpu()
        val_loss_sum+=loss_vt.detach().numpy()
        prediction_vt=prediction_v.cpu()
        prediction_v_np=prediction_vt.detach().numpy()
        for i in range(len(prediction_v_np)):
            if np.argmax(prediction_v_np[i])==label_np[i]:
                val_acc_sum+=1
            counter+=1
        loss_list.append(val_loss_sum)
        acc_list.append(val_acc_sum)
        counter_list.append(counter)

    return loss_list,acc_list,counter_list


def expand_image(input,expand_ratio=7):
    input=torch.repeat_interleave(input,expand_ratio,2)
    input=torch.repeat_interleave(input,expand_ratio,3)
    return input

def add_noise(input,norm,it,loader,bs):
    try:
        random_noise, labels = it.next()
        if random_noise.shape[0]!=bs:
            it = iter(loader)
            random_noise, labels = it.next()
    except StopIteration:
        it = iter(loader)
        random_noise, labels = it.next()
    random_noise=random_noise.cuda()
    random_noise=expand_image(random_noise)
    b,c,h,w=random_noise.shape
    f_rn=torch.reshape(random_noise,(b,c*h*w))
    noise_norms=torch.norm(f_rn,dim=1,keepdim=True)
    f_rn=norm*f_rn/noise_norms
    rn=torch.reshape(f_rn,(b,c,h,w))
    return rn+input,it

def main_ae(model,task_name):
    batch_size=args.batch_size
    device = torch.device('cuda')



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
    test_loader = IMAGENET_TEST_LOADER_PLACEHOLDER(test_aug)

    noise_list=[]
    for i in range(30):
        noise_list.append(i*0.05)

    loss_func = torch.nn.CrossEntropyLoss()

    result_acc_list=[]
    result_loss_list=[]
    result_counter_list=[]

    for stepv, (batch_xv,batch_yv) in enumerate(test_loader):
        if stepv==args.break_val:
            break
        batch_xv=batch_xv.cuda()
        batch_yv=batch_yv.cuda()
        if stepv%10==0:
            print(stepv)
            with open(current_folder+'/'+args.model_name+'_'+task_name+'_stdout'+".txt", "a") as std_out:
                std_out.write('step: '+str(stepv)+'\n')
                std_out.close()
        loss_list,acc_list,counter_list=attack(model,batch_xv,batch_yv,noise_list,loss_func)
        result_acc_list.append(acc_list)
        result_loss_list.append(loss_list)
        result_counter_list.append(counter_list)

    result_acc_list=np.asarray(result_acc_list)
    result_loss_list=np.asarray(result_loss_list)
    result_counter_list=np.asarray(result_counter_list)

    result_acc_list=np.sum(result_acc_list,0)
    result_loss_list=np.sum(result_loss_list,0)
    result_counter_list=np.sum(result_counter_list,0)

    result_acc_list=result_acc_list/result_counter_list
    result_loss_list=result_loss_list/result_counter_list

    print('acc list',result_acc_list)
    print('loss list',result_loss_list)
    with open(current_folder+'/'+args.model_name+'_'+task_name+'_stdout'+".txt", "a") as std_out:
        std_out.write('acc list'+str(result_acc_list)+'\n')
        std_out.write('loss list'+str(result_loss_list)+'\n')
        std_out.close()
    output_array=np.asarray([noise_list,result_acc_list,result_loss_list])
    np.save(current_folder+'/npout_'+args.model_name+'_'+task_name+'.npy',output_array)

def main_cifar(model,task_name):
    batch_size=args.batch_size
    device = torch.device('cuda')


    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_data=torchvision.datasets.CIFAR10(root='cifar_directory',transform=transforms_,download=True, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size
    )


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
    data_loader_val = IMAGENET_TEST_LOADER_PLACEHOLDER(test_aug)




    noise_list=[]
    for i in range(15):
        noise_list.append(i*50)

    loss_func = torch.nn.CrossEntropyLoss()

    result_acc_list=[]
    result_loss_list=[]

    for k in range(len(noise_list)):
        it = iter(test_loader)
        val_loss_sum=0
        val_acc_sum=0
        val_count=0
        for stepv, (batch_xv,batch_yv) in enumerate(data_loader_val):
            if stepv==args.break_val:
                break
            if batch_xv.shape[0]!=batch_size:
                break
            batch_xv=batch_xv.cuda()
            batch_xv,it=add_noise(batch_xv,noise_list[k],it,test_loader,batch_size)
            batch_yv=batch_yv.cuda()
            batch_yvt=batch_yv.cpu()
            labelv_np=batch_yvt.detach().numpy()
            val_count+=len(labelv_np)
            if args.model_name=='suf_vit_ti' or args.model_name=='improved4':
                prediction_v=model(batch_xv)
                prediction_v=prediction_v[0]
            else:
                prediction_v=model(batch_xv)

            loss_v = loss_func(prediction_v,batch_yv)
            loss_vt=loss_v.cpu()
            val_loss_sum+=loss_vt.detach().numpy()
            prediction_vt=prediction_v.cpu()
            prediction_v_np=prediction_vt.detach().numpy()
            for i in range(len(prediction_v_np)):
                if np.argmax(prediction_v_np[i])==labelv_np[i]:
                    val_acc_sum+=1
        print('norm: ',noise_list[k],
            ' val loss: ',val_loss_sum/val_count,
            ' val acc: ',val_acc_sum/val_count)

        result_acc_list.append(val_acc_sum/val_count)
        result_loss_list.append(val_loss_sum/val_count)

        with open(current_folder+'/'+args.model_name+'_'+task_name+'_stdout'+".txt", "a") as std_out:
            std_out.write('norm: '+str(noise_list[k])+' val loss: '+str(val_loss_sum/val_count)+' val acc: '+str(val_acc_sum/val_count)+'\n')
            std_out.write('\n')
            std_out.close()

    output_array=np.asarray([noise_list,result_acc_list,result_loss_list])
    np.save(current_folder+'/npout_'+args.model_name+'_'+task_name+'.npy',output_array)

if __name__ == "__main__":


    if args.model_name=='suf_vit_ti':
        model = suf_vit_ti()
        model = nn.DataParallel(model).cuda()
        checkpoint =  torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model.eval()


    task_name='ae'
    std_out=open(current_folder+'/'+args.model_name+'_'+task_name+'_stdout'+'.txt','w+')
    std_out.close()
    main_ae(model,task_name)

    task_name='cifar'
    std_out=open(current_folder+'/'+args.model_name+'_'+task_name+'_stdout'+'.txt','w+')
    std_out.close()
    main_cifar(model,task_name)


