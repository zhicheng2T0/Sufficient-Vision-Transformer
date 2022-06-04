'''
Code modified based on https://github.com/yitu-opensource/T2T-ViT, the corresponding research paper
"Token-to-Token ViT: Training Vision Transformer from Scratch on ImageNet" is available at
https://arxiv.org/abs/2101.11986.
'''

import torch
import torch.nn as nn
import argparse
import numpy as np
import math
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

from typing import Tuple

from data.randaugment import rand_augment_transform
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from timm.scheduler import create_scheduler
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

import time

model_name='aug_vit_small'
current_folder='.'


mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=10)

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Model(nn.Module):
    def __init__(self, info_list,token_num_list,inter_layer_dim,mlp_dim,patch_dim,
    img_size=32, tokens_type='performer', in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0,
                 drop_path_rate=0, norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.patch_dim=patch_dim
        self.flatten_dim=patch_dim*patch_dim*in_chans
        self.info_list=info_list

        num_token=1+int(img_size*img_size*in_chans/(self.flatten_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_token, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        self.patch_to_embedding=nn.Linear(self.flatten_dim, embed_dim)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward_features(self, x):
        B = x.shape[0]


        n, c, h, w = x.shape
        x = (
            x.unfold(2, self.patch_dim, self.patch_dim)
            .unfold(3, self.patch_dim, self.patch_dim)
            .contiguous()
        )
        x = x.view(n, c, -1, self.patch_dim ** 2)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, self.flatten_dim)

        x=self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        out_info=[]
        count=0
        for blk in self.blocks:
            if count in self.info_list:
                out_info.append(x)
            x = blk(x)
            count+=1


        return x[:, 0],out_info

    def forward(self, x):
        x,out_info = self.forward_features(x)
        x = self.head(x)
        return x,out_info


def reshape(x,k):
    unfold=nn.Unfold(kernel_size=(k, k), stride=(k, k))
    x=unfold(x)
    x=torch.transpose(x,2,1)
    return x

def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)


def naive_cross_entropy_loss(input: torch.Tensor,
                             target: torch.Tensor
                             ) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()

class MINE(nn.Module):
  def __init__(self,dim1,dim2,num1,num2,
          qkv_bias=False, qk_scale=None,attn_drop=0.,
          act_layer=nn.GELU, norm_layer=nn.LayerNorm,drop=0.):
    super().__init__()
    self.reduce_dim=30
    self.norm1 = norm_layer(dim1)
    self.norm2 = norm_layer(dim2)
    self.num1=num1
    self.num2=num2
    self.fc10=nn.Sequential(nn.Linear(dim1, self.reduce_dim),nn.GELU())
    self.fc11=nn.Sequential(nn.Linear(dim2, self.reduce_dim),nn.GELU())
    self.norm120=norm_layer(self.reduce_dim)
    self.norm121=norm_layer(self.reduce_dim)
    self.fc120=nn.Sequential(nn.Linear(self.reduce_dim, self.reduce_dim),nn.GELU())
    self.fc121=nn.Sequential(nn.Linear(self.reduce_dim, self.reduce_dim),nn.GELU())

    self.norm3=norm_layer(num1+num2)
    self.fc20=nn.Sequential(nn.Linear(num1+num2, 1),nn.GELU())

    self.fc3=nn.Sequential(norm_layer(self.reduce_dim),nn.Linear(self.reduce_dim, self.reduce_dim),nn.GELU())
    self.fc_out=nn.Sequential(norm_layer(self.reduce_dim),nn.Linear(self.reduce_dim, 1))

  def forward(self,input1,input2):
    input1 = self.fc10(self.norm1(input1)/self.num1)
    input2 = self.fc11(self.norm2(input2)/self.num2)
    input1=self.norm120(input1)
    input2=self.norm121(input2)
    input1 = self.fc120(input1)+input1
    input2 = self.fc121(input2)+input2

    input1=torch.transpose(input1,1,2)
    input2=torch.transpose(input2,1,2)
    cat=torch.cat([input1,input2],2)

    cat=torch.squeeze(self.fc20(self.norm3(cat)))
    cat=self.fc3(cat)
    output=self.fc_out(cat)
    return output

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


    return output

def reshape_y(y):
    new=torch.zeros(y.shape[0],10)
    yt=y.cpu()
    y_np=yt.detach().numpy()
    for i in range(len(y_np)):
        new[i][y_np[i]]=1
    new=new.cuda()
    return new

def main():
    valid_size=1/50000
    batch_size=64
    epochs=15
    token_num_=64
    k=4
    in_chans=3
    patch_dim_=4
    token_dim_=384
    num_classes_=10
    image_size_=32
    drop_rate=0.1
    mlp_ratio=4

    token_num_list=[token_dim_,token_dim_,token_dim_,token_dim_,
                    token_dim_,token_dim_,token_dim_,token_dim_,
                    token_dim_,token_dim_,token_dim_,token_dim_]
    inter_layer_dim=[]

    info_list=[0,2,5,7,8,9,10,11]

    model=Model(info_list=info_list,
                token_num_list=token_num_list,
                inter_layer_dim=inter_layer_dim,
                img_size=image_size_,
                mlp_dim=512,
                tokens_type='performer',
                in_chans=in_chans,
                num_classes=num_classes_,
                patch_dim=patch_dim_,
                embed_dim=token_dim_,
                depth=len(token_num_list),
                num_heads=6,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=drop_rate,
                attn_drop_rate=drop_rate,
                drop_path_rate=drop_rate,
                norm_layer=nn.LayerNorm,
                mlp_ratio=mlp_ratio,
                token_dim=token_dim_)
    model.load_state_dict(torch.load(current_folder+'/'+model_name+'/'+model_name))
    model=model.cuda()

    num0=token_num_
    dim0=token_dim_
    ori_dim=[num0,dim0]
    num_dim=[[[num0,dim0],[num0,dim0],[num0,dim0]],[[num0,dim0],[num0,dim0],[num0,dim0],[num0,dim0]]]


    for k in range(len(num_dim)):

        MINE_list=[]
        for i in range(len(num_dim[k])):
            MINE_=MINE(dim1=10,
                    dim2=num_dim[k][i][1],
                    num1=1,
                    num2=num_dim[k][i][0]+1,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0,
                    attn_drop=0)
            MINE_=MINE_.cuda()
            MINE_list.append(MINE_)



        MINE_param=[]
        for MINE_ in MINE_list:
            for p in MINE_.parameters():
                MINE_param.append(p)

        optimizer_mine=torch.optim.Adam(MINE_param, lr=0.0001)
        lambda1=lambda epoch:(epoch/4000) if epoch<4000 else 0.5*(math.cos((epoch-4000)/(100*1000-4000)*math.pi)+1)
        scheduler_mine=optim.lr_scheduler.LambdaLR(optimizer_mine,lr_lambda=lambda1)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset=torchvision.datasets.CIFAR10(root=current_folder+'/dataset',transform=transforms_,download=True, train=True)
        dataset_val=torchvision.datasets.CIFAR10(root=current_folder+'/dataset',transform=transforms_,download=True, train=True)
        test_data=torchvision.datasets.CIFAR10(root=current_folder+'/dataset_test',transform=transforms_,download=True, train=False)

        num_train = 50000
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)


        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, sampler=valid_sampler
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size
        )

        loss_func = torch.nn.CrossEntropyLoss()

        best_val_acc=0

        accumulate_num=1
        for epoch in range(epochs):
            train_acc_sum=0
            train_count=0

            accumulated_count=0
            start_time=time.time()
            for step, (batch_x,batch_y) in enumerate(train_loader):
                if k==0:
                    mine_sum_list=[0,0,0]
                    mine_count_list=[0,0,0]
                if k==1:
                    mine_sum_list=[0,0,0,0]
                    mine_count_list=[0,0,0,0]

                model.train()
                if batch_x.shape[0]!=batch_size:
                    break

                batch_x=batch_x.cuda()
                batch_y=batch_y.cuda()

                batch_x0=batch_x
                batch_y0=batch_y

                batch_y_1h=reshape_y(batch_y)
                batch_y_1h=torch.reshape(batch_y_1h,(batch_y_1h.shape[0],1,batch_y_1h.shape[1]))

                if accumulated_count==0:
                    optimizer_mine.zero_grad()

                prediction,info_list=model(batch_x0)
                ord_list=np.arange(prediction.shape[0])
                np.random.shuffle(ord_list)

                x0=info_list[0]
                x10=x0.clone().detach()[ord_list]
                x1=info_list[1]
                x11=x1.clone().detach()[ord_list]
                x2=info_list[2]
                x12=x2.clone().detach()[ord_list]
                x3=info_list[3]
                x13=x3.clone().detach()[ord_list]
                x4=info_list[4]
                x14=x4.clone().detach()[ord_list]
                x5=info_list[5]
                x15=x5.clone().detach()[ord_list]
                x6=info_list[6]
                x16=x6.clone().detach()[ord_list]
                x7=info_list[7]
                x17=x7.clone().detach()[ord_list]
                batch_y_1h1=batch_y_1h.clone().detach()[ord_list]

                if k==0:
                    m0p=MINE_list[0](batch_y_1h.detach(),x1.detach())
                    m0q=MINE_list[0](batch_y_1h1.detach(),x1.detach())

                    m1p=MINE_list[1](batch_y_1h.detach(),x2.detach())
                    m1q=MINE_list[1](batch_y_1h1.detach(),x2.detach())

                    m2p=MINE_list[2](batch_y_1h.detach(),x3.detach())
                    m2q=MINE_list[2](batch_y_1h1.detach(),x3.detach())

                    loss_mine0=calculate_mi(m0p,m0q)
                    loss_mine1=calculate_mi(m1p,m1q)
                    loss_mine2=calculate_mi(m2p,m2q)

                    loss_mine=-loss_mine0-loss_mine1-loss_mine2

                    loss_mine.backward()

                    optimizer_mine.step()
                    scheduler_mine.step()

                    mine_sum_list[0]+=loss_mine0.cpu().detach().numpy()
                    mine_count_list[0]+=1
                    mine_sum_list[1]+=loss_mine1.cpu().detach().numpy()
                    mine_count_list[1]+=1
                    mine_sum_list[2]+=loss_mine2.cpu().detach().numpy()
                    mine_count_list[2]+=1

                if k==1:
                    m3p=MINE_list[0](batch_y_1h.detach(),x4.detach())
                    m3q=MINE_list[0](batch_y_1h1.detach(),x4.detach())

                    m4p=MINE_list[1](batch_y_1h.detach(),x5.detach())
                    m4q=MINE_list[1](batch_y_1h1.detach(),x5.detach())

                    m5p=MINE_list[2](batch_y_1h.detach(),x6.detach())
                    m5q=MINE_list[2](batch_y_1h1.detach(),x6.detach())

                    m6p=MINE_list[3](batch_y_1h.detach(),x7.detach())
                    m6q=MINE_list[3](batch_y_1h1.detach(),x7.detach())

                    loss_mine3=calculate_mi(m3p,m3q)
                    loss_mine4=calculate_mi(m4p,m4q)
                    loss_mine5=calculate_mi(m5p,m5q)
                    loss_mine6=calculate_mi(m6p,m6q)

                    loss_mine=-loss_mine3-loss_mine4-loss_mine5-loss_mine6

                    loss_mine.backward()

                    optimizer_mine.step()
                    scheduler_mine.step()

                    mine_sum_list[0]+=loss_mine3.cpu().detach().numpy()
                    mine_count_list[0]+=1
                    mine_sum_list[1]+=loss_mine4.cpu().detach().numpy()
                    mine_count_list[1]+=1
                    mine_sum_list[2]+=loss_mine5.cpu().detach().numpy()
                    mine_count_list[2]+=1
                    mine_sum_list[3]+=loss_mine6.cpu().detach().numpy()
                    mine_count_list[3]+=1


                if step==int(num_train/batch_size)-100:

                    if k==0:
                        mine_sum_listv=[0,0,0]
                        mine_count_listv=[0,0,0]
                    if k==1:
                        mine_sum_listv=[0,0,0,0]
                        mine_count_listv=[0,0,0,0]

                    for stepv, (batch_xv,batch_yv) in enumerate(test_loader):
                        batch_xv=batch_xv.cuda()
                        prediction,info_list=model(batch_xv)
                        ord_list=np.arange(prediction.shape[0])
                        np.random.shuffle(ord_list)

                        batch_yv_1h=reshape_y(batch_yv)
                        batch_yv_1h=torch.reshape(batch_yv_1h,(batch_yv_1h.shape[0],1,batch_yv_1h.shape[1]))

                        x0=info_list[0]
                        x10=x0.clone().detach()[ord_list]
                        x1=info_list[1]
                        x11=x1.clone().detach()[ord_list]
                        x2=info_list[2]
                        x12=x2.clone().detach()[ord_list]
                        x3=info_list[3]
                        x13=x3.clone().detach()[ord_list]
                        x4=info_list[4]
                        x14=x4.clone().detach()[ord_list]
                        x5=info_list[5]
                        x15=x5.clone().detach()[ord_list]
                        x6=info_list[6]
                        x16=x6.clone().detach()[ord_list]
                        x7=info_list[7]
                        x17=x7.clone().detach()[ord_list]
                        batch_yv_1h1=batch_yv_1h.clone().detach()[ord_list]


                        if k==0:
                            m0p=MINE_list[0](batch_yv_1h.detach(),x1.detach())
                            m0q=MINE_list[0](batch_yv_1h1.detach(),x1.detach())

                            m1p=MINE_list[1](batch_yv_1h.detach(),x2.detach())
                            m1q=MINE_list[1](batch_yv_1h1.detach(),x2.detach())

                            m2p=MINE_list[2](batch_yv_1h.detach(),x3.detach())
                            m2q=MINE_list[2](batch_yv_1h1.detach(),x3.detach())

                            loss_mine0=calculate_mi(m0p,m0q)
                            loss_mine1=calculate_mi(m1p,m1q)
                            loss_mine2=calculate_mi(m2p,m2q)

                            mine_sum_listv[0]+=loss_mine0.cpu().detach().numpy()
                            mine_count_listv[0]+=1
                            mine_sum_listv[1]+=loss_mine1.cpu().detach().numpy()
                            mine_count_listv[1]+=1
                            mine_sum_listv[2]+=loss_mine2.cpu().detach().numpy()
                            mine_count_listv[2]+=1

                        if k==1:
                            m3p=MINE_list[0](batch_yv_1h.detach(),x4.detach())
                            m3q=MINE_list[0](batch_yv_1h1.detach(),x4.detach())

                            m4p=MINE_list[1](batch_yv_1h.detach(),x5.detach())
                            m4q=MINE_list[1](batch_yv_1h1.detach(),x5.detach())

                            m5p=MINE_list[2](batch_yv_1h.detach(),x6.detach())
                            m5q=MINE_list[2](batch_yv_1h1.detach(),x6.detach())

                            m6p=MINE_list[3](batch_yv_1h.detach(),x7.detach())
                            m6q=MINE_list[3](batch_yv_1h1.detach(),x7.detach())

                            loss_mine3=calculate_mi(m3p,m3q)
                            loss_mine4=calculate_mi(m4p,m4q)
                            loss_mine5=calculate_mi(m5p,m5q)
                            loss_mine6=calculate_mi(m6p,m6q)

                            mine_sum_listv[0]+=loss_mine3.cpu().detach().numpy()
                            mine_count_listv[0]+=1
                            mine_sum_listv[1]+=loss_mine4.cpu().detach().numpy()
                            mine_count_listv[1]+=1
                            mine_sum_listv[2]+=loss_mine5.cpu().detach().numpy()
                            mine_count_listv[2]+=1
                            mine_sum_listv[3]+=loss_mine6.cpu().detach().numpy()
                            mine_count_listv[3]+=1

                    if k==0:
                        print('epoch: ',epoch)
                        print('train: term 0: ',mine_sum_list[0]/mine_count_list[0],
                                ' term 1: ',mine_sum_list[1]/mine_count_list[1],
                                ' term 2: ',mine_sum_list[2]/mine_count_list[2])
                        print('test: term 0: ',mine_sum_listv[0]/mine_count_listv[0],
                                ' term 1: ',mine_sum_listv[1]/mine_count_listv[1],
                                ' term 2: ',mine_sum_listv[2]/mine_count_listv[2])
                        with open(current_folder+'/'+model_name+'_info_y_stdout'+".txt", "a") as std_out:
                            std_out.write('epoch: '+str(epoch)+'\n')
                            std_out.write('train: term 0: '+str(mine_sum_list[0]/mine_count_list[0])+
                                    ' term 1: '+str(mine_sum_list[1]/mine_count_list[1])+
                                    ' term 2: '+str(mine_sum_list[2]/mine_count_list[2])+'\n')
                            std_out.write('test: term 0: '+str(mine_sum_listv[0]/mine_count_listv[0])+
                                    ' term 1: '+str(mine_sum_listv[1]/mine_count_listv[1])+
                                    ' term 2: '+str(mine_sum_listv[2]/mine_count_listv[2])+'\n')
                            std_out.write('\n')
                            std_out.close()
                    if k==1:
                        print('epoch: ',epoch)
                        print('train: term 3: ',mine_sum_list[0]/mine_count_list[0],
                                ' term 4: ',mine_sum_list[1]/mine_count_list[1],
                                ' term 5: ',mine_sum_list[2]/mine_count_list[2],
                                ' term 6: ',mine_sum_list[3]/mine_count_list[3])
                        print('test: term 3: ',mine_sum_listv[0]/mine_count_listv[0],
                                ' term 4: ',mine_sum_listv[1]/mine_count_listv[1],
                                ' term 5: ',mine_sum_listv[2]/mine_count_listv[2],
                                ' term 6: ',mine_sum_listv[3]/mine_count_listv[3])
                        with open(current_folder+'/'+model_name+'_info_y_stdout'+".txt", "a") as std_out:
                            std_out.write('epoch: '+str(epoch)+'\n')
                            std_out.write('train: term 3: '+str(mine_sum_list[0]/mine_count_list[0])+
                                    ' term 4: '+str(mine_sum_list[1]/mine_count_list[1])+
                                    ' term 5: '+str(mine_sum_list[2]/mine_count_list[2])+
                                    ' term 6: '+str(mine_sum_list[3]/mine_count_list[3])+'\n')
                            std_out.write('test: term 3: '+str(mine_sum_listv[0]/mine_count_listv[0])+
                                    ' term 4: '+str(mine_sum_listv[1]/mine_count_listv[1])+
                                    ' term 5: '+str(mine_sum_listv[2]/mine_count_listv[2])+
                                    ' term 6: '+str(mine_sum_listv[3]/mine_count_listv[3])+'\n')
                            std_out.write('\n')
                            std_out.close()


                batch_x0=None
                batch_y0=None


if __name__=='__main__':

    std_out=open(current_folder+'/'+model_name+'_info_y_stdout'+'.txt','w+')
    std_out.close()

    if not os.path.exists(current_folder+'/'+model_name):
        os.mkdir(current_folder+'/'+model_name)

    main()