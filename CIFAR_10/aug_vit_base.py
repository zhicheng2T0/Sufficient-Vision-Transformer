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

model_name='aug_vit_base'
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
    def __init__(self, token_num_list,inter_layer_dim,mlp_dim,patch_dim,
    img_size=32, tokens_type='performer', in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0, attn_drop_rate=0,
                 drop_path_rate=0, norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.patch_dim=patch_dim
        self.flatten_dim=patch_dim*patch_dim*in_chans

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

        count=0
        for blk in self.blocks:
            x = blk(x)
            count+=1

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def reshape_y(y):
    new=torch.zeros(y.shape[0],10)
    yt=y.cpu()
    y_np=yt.detach().numpy()
    for i in range(len(y_np)):
        new[i][y_np[i]]=1
    return new

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


def main():
    valid_size=1/50000
    batch_size=64
    epochs=150
    token_num_=64
    k=4
    in_chans=3
    patch_dim_=4
    token_dim_=768
    num_classes_=10
    image_size_=32
    drop_rate=0.1
    mlp_ratio=4

    token_num_list=[token_dim_,token_dim_,token_dim_,token_dim_,
                    token_dim_,token_dim_,token_dim_,token_dim_,
                    token_dim_,token_dim_,token_dim_,token_dim_]
    inter_layer_dim=[]

    model=Model(token_num_list=token_num_list,
                inter_layer_dim=inter_layer_dim,
                img_size=image_size_,
                mlp_dim=512,
                tokens_type='performer',
                in_chans=in_chans,
                num_classes=num_classes_,
                patch_dim=patch_dim_,
                embed_dim=token_dim_,
                depth=len(token_num_list),
                num_heads=12,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=drop_rate,
                attn_drop_rate=drop_rate,
                drop_path_rate=drop_rate,
                norm_layer=nn.LayerNorm,
                mlp_ratio=mlp_ratio,
                token_dim=token_dim_)
    model=model.cuda()
    optimizer_network=torch.optim.Adam(model.parameters(), lr=0.0001)
    lambda1=lambda epoch:(epoch/4000) if epoch<4000 else 0.5*(math.cos((epoch-4000)/(100*1000-4000)*math.pi)+1)
    scheduler_network=optim.lr_scheduler.LambdaLR(optimizer_network,lr_lambda=lambda1)

    mean = (0.485, 0.456, 0.406)
    aa_params = dict(
            translate_const=int(32 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset=torchvision.datasets.CIFAR10(root=current_folder+'/dataset',transform=transform_train,download=True, train=True)
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

    for epoch in range(epochs):
        for step, (batch_x,batch_y) in enumerate(train_loader):
            if batch_x.shape[0]!=batch_size:
                break
            model.train()
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()

            batch_x, batch_y_t = mixup_fn(batch_x, batch_y)

            prediction=model(batch_x)
            loss_net = torch.sum(-batch_y_t * F.log_softmax(prediction, dim=-1), dim=-1).mean()

            optimizer_network.zero_grad()
            loss_net.backward()
            optimizer_network.step()

            scheduler_network.step()

            if step==int(num_train/batch_size)-100:
                model.eval()
                train_loss_sum=0
                loss=loss_func(prediction,batch_y)
                losst=loss.cpu()
                loss_np=losst.detach().numpy()
                train_loss_sum+=loss_np

                train_acc_sum=0
                predictiont=prediction.cpu()
                prediction_np=predictiont.detach().numpy()
                batch_yt=batch_y.cpu()
                label_np=batch_yt.detach().numpy()
                for i in range(len(prediction_np)):
                    if np.argmax(prediction_np[i])==label_np[i]:
                        train_acc_sum+=1

                train_count=0
                train_count=batch_size

                #validation result
                val_loss_sum=0
                val_acc_sum=0
                val_count=0
                for stepv, (batch_xv,batch_yv) in enumerate(test_loader):
                    batch_xv=batch_xv.cuda()
                    batch_yv=batch_yv.cuda()
                    batch_yvt=batch_yv.cpu()
                    labelv_np=batch_yvt.detach().numpy()
                    val_count+=len(labelv_np)
                    prediction_v=model(batch_xv)

                    loss_v = loss_func(prediction_v,batch_yv)
                    loss_vt=loss_v.cpu()
                    val_loss_sum+=loss_vt.detach().numpy()
                    prediction_vt=prediction_v.cpu()
                    prediction_v_np=prediction_vt.detach().numpy()
                    for i in range(len(prediction_v_np)):
                        if np.argmax(prediction_v_np[i])==labelv_np[i]:
                            val_acc_sum+=1

                if val_acc_sum/val_count>best_val_acc:
                    best_val_acc=val_acc_sum/val_count
                    torch.save(model.state_dict(), current_folder+'/'+model_name+'/'+model_name+'_best')

                print('epoch: ',epoch,
                    ' step: ',step,
                    ' train loss: ',train_loss_sum/train_count,
                    ' train acc: ',train_acc_sum/train_count,
                    ' val loss: ',val_loss_sum/val_count,
                    ' val acc: ',val_acc_sum/val_count,
                    'best val acc: ',best_val_acc)

                with open(current_folder+'/'+model_name+'_stdout'+".txt", "a") as std_out:
                    std_out.write('epoch: '+str(epoch)+' step: '+str(step)+' train loss: '+str(train_loss_sum/train_count)+' train acc: '+str(train_acc_sum/train_count)+' val loss: '+str(val_loss_sum/val_count)+' val acc: '+str(val_acc_sum/val_count)+' best val acc: '+str(best_val_acc)+'\n')
                    std_out.write('\n')
                    std_out.close()
                torch.save(model.state_dict(), current_folder+'/'+model_name+'/'+model_name)

def try_model():
    valid_size=1/50000
    batch_size=64
    epochs=150
    token_num_=64
    k=4
    in_chans=3
    patch_dim_=4
    token_dim_=768
    num_classes_=10
    image_size_=32
    drop_rate=0.1
    mlp_ratio=4

    token_num_list=[token_dim_,token_dim_,token_dim_,token_dim_,
                    token_dim_,token_dim_,token_dim_,token_dim_,
                    token_dim_,token_dim_,token_dim_,token_dim_]
    inter_layer_dim=[]

    model=Model(token_num_list=token_num_list,
                inter_layer_dim=inter_layer_dim,
                img_size=image_size_,
                mlp_dim=512,
                tokens_type='performer',
                in_chans=in_chans,
                num_classes=num_classes_,
                patch_dim=patch_dim_,
                embed_dim=token_dim_,
                depth=len(token_num_list),
                num_heads=12,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=drop_rate,
                attn_drop_rate=drop_rate,
                drop_path_rate=drop_rate,
                norm_layer=nn.LayerNorm,
                mlp_ratio=mlp_ratio,
                token_dim=token_dim_)

    model_param=[]
    param_count=0
    for p in model.parameters():
        model_param.append(p)
        shape_tupple=p.detach().numpy().shape
        start=1
        for i in range(len(shape_tupple)):
            start=start*shape_tupple[i]
        param_count+=start
    print('param_count: ',param_count)

if __name__=='__main__':
    std_out=open(current_folder+'/'+model_name+'_stdout'+'.txt','w+')
    std_out.close()

    if not os.path.exists(current_folder+'/'+model_name):
        os.mkdir(current_folder+'/'+model_name)

    main()



