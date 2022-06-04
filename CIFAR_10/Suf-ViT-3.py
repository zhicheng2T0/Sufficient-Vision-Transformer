import torch
from torch import nn
import math
from network.trunc_norm import trunc_normal_
from network.token_performer import Token_performer
import numpy as np

from functools import partial
from einops import rearrange
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
import time
from data.random_erasing import RandomErasing

from einops import rearrange
from data.randaugment import rand_augment_transform
from util.mixup import Mixup
import torch.nn.functional as F
mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=10)

current_folder='.'
dataset_dir='cifar 10 data directory'
model_name='suf_vit_ti'

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_inter(nn.Module):
    def __init__(self,num_ori,num_new, dim_old,dim_new, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_ori=num_ori
        self.num_new=num_new
        self.dim_old=dim_old
        self.dim_new=dim_new
        self.num_heads = num_heads
        head_dim = dim_new // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce_num=nn.Sequential(nn.Linear(num_ori,num_new*4),nn.ReLU(),nn.Linear(num_new*4,num_new))
        self.increase_dimq=nn.Linear(self.dim_old,self.dim_new)

        self.kv = nn.Linear(self.dim_old, self.dim_new * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_new, self.dim_new)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = nn.LayerNorm(self.dim_old)


    def forward(self, x):
        B, N, C = x.shape

        x=self.norm1(x)

        q=torch.transpose(self.reduce_num(torch.transpose(x,1,2)),1,2)
        q=self.increase_dimq(q)
        q=torch.reshape(q,(B,self.num_heads,self.num_new,self.dim_new//self.num_heads))

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.dim_new // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_new, self.dim_new)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Block2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_ = x + self.drop_path(self.attn(self.norm1(x)))
        x = x_ + self.drop_path(self.mlp(self.norm2(x_)))
        x2 = x_ + self.drop_path(self.mlp2(self.norm2(x_)))
        return x,x2

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



class Transformer(nn.Module):
    def __init__(self, norm_layer,base_dim, depth, heads, mlp_ratio,change_layer,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        self.depth=depth

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        if change_layer!=None:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.1,
                    norm_layer=norm_layer
                )
                if i not in [change_layer]
                else
                Block2(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.1,
                    norm_layer=norm_layer
                )
                for i in range(depth)])

    def forward(self, x, cls_tokens,inter_start=None):
        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        counter=0
        for blk in self.blocks:
            if inter_start!=None and inter_start==counter:
                x_0=x
                x_00,x_01=blk(x)
                x=x_00
                break
            x = blk(x)
            counter+=1

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]

        if inter_start!=None:
            return x, cls_tokens,x_0,x_00,x_01
        else:
            return x, cls_tokens

class my_conv(nn.Module):
    def __init__(self, in_feature, out_feature,kernel_size,padding, stride,groups):
        super(my_conv, self).__init__()

        self.kernel_size=kernel_size
        self.padding=padding
        self.groups=groups#12
        self.stride=stride

        self.unfolding=torch.nn.Unfold(kernel_size=self.kernel_size,
                            stride=self.stride)

        weight1 = torch.Tensor(1,1,in_feature, kernel_size*kernel_size)
        self.weight1 = nn.Parameter(weight1, requires_grad=True)

        weight2 = torch.Tensor(1,1,in_feature, kernel_size*kernel_size)
        self.weight2 = nn.Parameter(weight2, requires_grad=True)

        bias = torch.Tensor(1,out_feature,1,1)
        self.bias = nn.Parameter(bias, requires_grad=True)

        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight1, 1.0)
        nn.init.constant_(self.weight2, 1.0)

        self.order_list=[]
        for i in range(in_feature):
            self.order_list.append(i)
            self.order_list.append(i+in_feature)



    def forward(self, x):

        x=torch.nn.functional.pad(input=x,
                                    pad=(self.padding,self.padding,self.padding,self.padding),
                                    mode='constant',
                                    value=0)
        x=self.unfolding(x)
        s0,s1,s2=x.shape
        x=rearrange(x,'b (d e) c -> b c d e',d=s1//(self.kernel_size*self.kernel_size),e=self.kernel_size*self.kernel_size)
        res1=x*self.weight1
        res1=torch.sum(res1,3)

        res2=x*self.weight2
        res2=torch.sum(res2,3)

        x=torch.cat([res1,res2],2)
        ss0,ss1,ss2=x.shape
        x=rearrange(x,'b (h w) c -> b c h w',h=int(math.sqrt(ss1)),w=int(math.sqrt(ss1)))
        x=x[:,self.order_list,:,:]
        x=x+self.bias
        return x


class my_conv2(nn.Module):
    def __init__(self, in_feature, out_feature,kernel_size,padding, stride):
        super(my_conv2, self).__init__()

        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride

        self.unfolding=torch.nn.Unfold(kernel_size=self.kernel_size,
                            stride=self.stride)
        self.fc=torch.nn.Linear(in_feature*kernel_size*kernel_size,out_feature)

        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.fc.weight, 1.0)

    def forward(self, x):

        x=torch.nn.functional.pad(input=x,
                                    pad=(self.padding,self.padding,self.padding,self.padding),
                                    mode='constant',
                                    value=0)
        x=torch.transpose(self.unfolding(x),1,2)
        x=self.fc(x)
        x=rearrange(x,'b (h w) c -> b c h w',h=int(math.sqrt(x.shape[1])),w=int(math.sqrt(x.shape[1])))


        return x

class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token



class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_inter(nn.Module):
    def __init__(self, num_old, num_new, dim_old, dim_new):
        super(fc_inter, self).__init__()

        self.num_old=num_old
        self.num_new=num_new
        self.dim_old=dim_old
        self.dim_new=dim_new

        self.c_dim_old=int(dim_old/4)
        self.c_dim_new=int(dim_new/4)


        self.channel0_0=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel0_1=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel1_0=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel1_1=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel2_0=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel2_1=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel3_0=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel3_1=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())

        self.fc = nn.Linear(dim_old, dim_new)

    def forward(self, x, cls_token):
        x=torch.transpose(x,1,2)
        x0=x[:,self.c_dim_old*0:self.c_dim_old*1,:]
        x1=x[:,self.c_dim_old*1:self.c_dim_old*2,:]
        x2=x[:,self.c_dim_old*2:self.c_dim_old*3,:]
        x3=x[:,self.c_dim_old*3:self.c_dim_old*4,:]

        x0_0=self.channel0_0(x0)
        x0_1=self.channel0_1(x0)
        x1_0=self.channel1_0(x1)
        x1_1=self.channel1_1(x1)
        x2_0=self.channel0_0(x2)
        x2_1=self.channel0_1(x2)
        x3_0=self.channel1_0(x3)
        x3_1=self.channel1_1(x3)

        x=torch.transpose(torch.cat([x0_0,x0_1,x1_0,x1_1,x2_0,x2_1,x3_0,x3_1],1),1,2)

        if cls_token!=None:
            cls_token = self.fc(cls_token)
            return x, cls_token
        else:
            return x


class sufvit(nn.Module):
    def __init__(self, head_list, token_num_list, token_dim_list, depth_list,
                    change_layer_list, inter_layer_num, inter_layer_dim, stride, base_dims,
                 mlp_ratio, img_size=224,patch_size=16, num_classes=10, in_chans=3,
                 attn_drop_rate=0, drop_rate=0, drop_path_rate=0.1,qkv_bias=False,qk_scale=None,
                 norm_layer=nn.LayerNorm,classifier='token',init = ''):

        super().__init__()


        self.classifier = classifier
        self.num_classes = num_classes

        self.token_num_list=token_num_list
        self.inter_layer_num=inter_layer_num
        self.token_dim_list=token_dim_list
        self.inter_layer_dim=inter_layer_dim
        self.change_layer_list=change_layer_list

        self.head_list=head_list
        self.depth_list=depth_list

        depth=len(self.token_num_list)


        total_block = sum(self.depth_list)
        padding = 0
        block_idx = 0
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * self.head_list[0],
                                          patch_size, stride, padding)

        width = math.floor(
            (img_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * self.head_list[0], int(np.sqrt(self.token_num_list[0])),int(np.sqrt(self.token_num_list[0]))),
            requires_grad=True
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * self.head_list[0]),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        self.inters = nn.ModuleList([])

        for stage in range(len(self.depth_list)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + self.depth_list[stage])]
            block_idx += self.depth_list[stage]

            self.transformers.append(
                Transformer(norm_layer,base_dims[stage],
                            self.depth_list[stage], self.head_list[stage],
                            mlp_ratio,self.change_layer_list[stage],
                            drop_rate, attn_drop_rate,drop_path_prob)
            )
            if stage < len(self.head_list) - 1:
                self.pools.append(
                            fc_inter(self.inter_layer_num[stage][0]+1,
                                        self.inter_layer_num[stage][1],
                                        self.inter_layer_dim[stage][0],
                                        self.inter_layer_dim[stage][1],
                                        )
                )
                self.inters.append(
                            fc_inter(self.inter_layer_num[stage][0]+1,
                                        self.inter_layer_num[stage][2],
                                        self.inter_layer_dim[stage][0],
                                        self.inter_layer_dim[stage][1],
                                        )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * self.head_list[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * self.head_list[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * self.head_list[-1], num_classes)
        else:
            self.head = nn.Identity()

        if init == 'ortho':
            self.apply(self._init_weights)
        elif init == 'trunc':
            self.apply(self._init_weights_trunc)
        elif init == 'constant':
            self.apply(self._init_weights_constant)
        else:
            raise RuntimeError("not support init type: {}".format(init))
        trunc_normal_(self.pos_embed, std=.02)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                trunc_normal_(m.bias, std=1e-6)

    def _init_weights_trunc(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                trunc_normal_(m.bias, std=1e-6)

    def _init_weights_constant(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        x=rearrange(x,'b c h w -> b (h w) c')
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        x_start_list=[]
        x_0_list=[]
        x_1_list=[]
        for stage in range(len(self.change_layer_list)-1):
            x, cls_tokens, x_start,x_0,x_1= self.transformers[stage](x, cls_tokens,self.change_layer_list[stage])
            x_0, cls_tokens = self.pools[stage](x_0, cls_tokens)
            x_1 = self.inters[stage](x_1,None)
            x_start_list.append(x_start)
            x_0_list.append(x_0)
            x_1_list.append(x_1)
            x=x_0
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens,x_start_list,x_0_list,x_1_list

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not 'head' in name:
                param.requires_grad = False

    def forward(self, x):
        cls_token,x_start_list,x_0_list,x_1_list = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token,x_start_list,x_0_list,x_1_list


class MINE(nn.Module):
  def __init__(self,dim1,dim2,num1,num2,
          qkv_bias=False, qk_scale=None,attn_drop=0.,
          act_layer=nn.GELU, norm_layer=nn.LayerNorm,drop=0.):
    super().__init__()
    self.reduce_dim=30
    self.norm1 = norm_layer(dim1)
    self.norm2 = norm_layer(dim2)
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
    input1 = self.fc10(self.norm1(input1))
    input2 = self.fc11(self.norm2(input2))
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

def scale_positional_embedding(posemb, new_posemb):
    ntok_new = new_posemb.size(1)
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = nn.functional.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)



def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def naive_cross_entropy_loss(input: torch.Tensor,
                             target: torch.Tensor
                             ) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


def reshape(x,k):
    unfold=nn.Unfold(kernel_size=(k, k), stride=(k, k))
    x=unfold(x)
    x=torch.transpose(x,2,1)
    return x

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

def calculate_loss_mine(p_og,q_og,p_ig,q_ig):
    term3=calculate_mi(p_og,q_og)
    term4=calculate_mi(p_ig,q_ig)
    output=-term3-term4
    return output

def reshape_y(y):
    new=torch.zeros(y.shape[0],10)
    yt=y.cpu()
    y_np=yt.detach().numpy()
    for i in range(len(y_np)):
        new[i][y_np[i]]=1
    return new






def main():
    valid_size=1/50000
    batch_size=128
    accumulate_num=4
    epochs=160
    token_num_=64
    k=4
    in_chans=3
    patch_dim_=4
    token_dim_=512
    num_classes_=10
    image_size_=32
    drop_rate=0.1

    head_list=[2,4,8]
    mlp_ratio=4

    num0=225
    num1=64
    num2=16
    token_num_list=[num0,num1,num2]
    dim0=96
    dim1=192
    dim2=384
    token_dim_list=[dim0,dim1,dim2]
    depth_list=[7,1,4]

    change_layer_list=[6,0,None]

    inter_layer_num_=[(num0,num1,32),(num1,num2,10)]
    inter_layer_dim_=[(dim0,dim1),(dim1,dim2)]

    model = sufvit(
        head_list=head_list,
        token_num_list=token_num_list,
        token_dim_list=token_dim_list,
        depth_list=depth_list,
        change_layer_list=change_layer_list,
        inter_layer_num=inter_layer_num_,
        inter_layer_dim=inter_layer_dim_,
        img_size=32,
        patch_size=4,
        stride=2,
        base_dims=[48, 48, 48],
        mlp_ratio=mlp_ratio,
        init = 'constant',
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    model=model.cuda()


    mine_heads=1
    MINE_OG_list=[]
    for i in range(len(inter_layer_dim_)):
        MINE_OG=MINE(dim1=inter_layer_dim_[i][1],
                dim2=inter_layer_dim_[i][1],
                num1=inter_layer_num_[i][1],
                num2=inter_layer_num_[i][2],
                qkv_bias=False,
                qk_scale=None,
                drop=0,
                attn_drop=0)
        MINE_OG=MINE_OG.cuda()
        MINE_OG_list.append(MINE_OG)

    MINE_IG_list=[]
    for i in range(len(inter_layer_dim_)):
        MINE_IG=MINE(dim1=inter_layer_dim_[i][0],
                dim2=inter_layer_dim_[i][1],
                num1=inter_layer_num_[i][0]+1,
                num2=inter_layer_num_[i][2],
                qkv_bias=False,
                qk_scale=None,
                drop=0,
                attn_drop=0)
        MINE_IG=MINE_IG.cuda()
        MINE_IG_list.append(MINE_IG)


    MINE_param=[]
    for model_mineog in MINE_OG_list:
        for p in model_mineog.parameters():
            MINE_param.append(p)
    for model_mineig in MINE_IG_list:
        for p in model_mineig.parameters():
            MINE_param.append(p)

    optimizer_network = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    optimizer_mine = torch.optim.AdamW(MINE_param, lr=0.0005, weight_decay=0.05)
    lambda1=lambda epoch:(epoch/4000) if epoch<4000 else 0.5*(math.cos((epoch-4000)/(100*1000-4000)*math.pi)+1)
    scheduler_network=optim.lr_scheduler.LambdaLR(optimizer_network,lr_lambda=lambda1)
    scheduler_mine=optim.lr_scheduler.LambdaLR(optimizer_mine,lr_lambda=lambda1)

    mean = (0.485, 0.456, 0.406)
    aa_params = dict(
            translate_const=int(32 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        rand_augment_transform('rand-m9-mstd0.5-inc1', aa_params),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(0.25, mode='pixel', max_count=1, num_splits=0, device='cpu')
    ])


    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset=torchvision.datasets.CIFAR10(root=dataset_dir,transform=transform_train,download=True, train=True)
    dataset_val=torchvision.datasets.CIFAR10(root=dataset_dir,transform=transforms_,download=True, train=True)
    test_data=torchvision.datasets.CIFAR10(root=dataset_dir,transform=transforms_,download=True, train=False)

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

    change_indexes=[5,10,15,40,60,80,101]
    change_constant=[0.8,14/28,12/28,10/28,8/28,7.7/28,7.3/28]
    current_change_index=0
    ratio=1

    best_val_acc=0

    for epoch in range(epochs):

        if i==change_indexes[current_change_index]:
            ratio=change_constant[current_change_index]
            current_change_index+=1

        train_acc_sum=0
        train_count=0

        accumulated_count=0
        start_time=time.time()
        for step, (batch_x,batch_y) in enumerate(train_loader):
            model.train()
            if batch_x.shape[0]!=batch_size:
                break

            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()


            alpha_=30/30
            beta1=(1/30)
            beta2=(1/30)
            w1=1
            w2=1
            w3=1

            batch_x, batch_y_t = mixup_fn(batch_x, batch_y)

            if accumulated_count==0:
                optimizer_network.zero_grad()
                optimizer_mine.zero_grad()

            prediction,x_start_list,x_0list,x_1list=model(batch_x)
            x_0=x_start_list[0]
            x_00=x_0list[0]
            x_01=x_1list[0]
            x_1=x_start_list[1]
            x_10=x_0list[1]
            x_11=x_1list[1]
            ord_list=np.arange(prediction.shape[0])
            np.random.shuffle(ord_list)

            prediction1=prediction.clone().detach()[ord_list]

            x1_0=x_0.clone().detach()[ord_list]
            x1_00=x_00.clone().detach()[ord_list]
            x1_01=x_01.clone().detach()[ord_list]

            x1_1=x_1.clone().detach()[ord_list]
            x1_10=x_10.clone().detach()[ord_list]
            x1_11=x_11.clone().detach()[ord_list]




            m1_ogp=MINE_OG_list[0](x_00,x_01)
            m1_ogq=MINE_OG_list[0](x1_00,x_01)

            m1_igp=MINE_IG_list[0](x_0,x_01)
            m1_igq=MINE_IG_list[0](x1_0,x_01)

            m2_ogp=MINE_OG_list[1](x_10,x_11)
            m2_ogq=MINE_OG_list[1](x1_10,x_11)

            m2_igp=MINE_IG_list[1](x_1,x_11)
            m2_igq=MINE_IG_list[1](x1_1,x_11)



            loss_net1=calculate_loss_network_mine(
                            p_og=m1_ogp,
                            q_og=m1_ogq,
                            p_ig=m1_igp,
                            q_ig=m1_igq,
                            beta=beta1)
            loss_net2=calculate_loss_network_mine(
                            p_og=m2_ogp,
                            q_og=m2_ogq,
                            p_ig=m2_igp,
                            q_ig=m2_igq,
                            beta=beta2)
            loss = torch.sum(-batch_y_t * F.log_softmax(prediction, dim=-1), dim=-1).mean()
            loss_net=loss+w1*loss_net1+w2*loss_net2

            if accumulate_num>1:
                loss_net.backward(retain_graph=True)
            else:
                loss_net.backward()

            m1_ogp=MINE_OG_list[0](x_00.detach(),x_01.detach())
            m1_ogq=MINE_OG_list[0](x1_00.detach(),x_01.detach())

            m1_igp=MINE_IG_list[0](x_0.detach(),x_01.detach())
            m1_igq=MINE_IG_list[0](x1_0.detach(),x_01.detach())

            m2_ogp=MINE_OG_list[1](x_10.detach(),x_11.detach())
            m2_ogq=MINE_OG_list[1](x1_10.detach(),x_11.detach())

            m2_igp=MINE_IG_list[1](x_1.detach(),x_11.detach())
            m2_igq=MINE_IG_list[1](x1_1.detach(),x_11.detach())



            loss_mine1=calculate_loss_mine(
                                        p_og=m1_ogp,
                                        q_og=m1_ogq,
                                        p_ig=m1_igp,
                                        q_ig=m1_igq)
            loss_mine2=calculate_loss_mine(
                                        p_og=m2_ogp,
                                        q_og=m2_ogq,
                                        p_ig=m2_igp,
                                        q_ig=m2_igq)
            loss_mine=loss_mine1+loss_mine2

            if accumulate_num>1:
                if accumulated_count+1==accumulate_num:
                    loss_mine.backward()
                else:
                    loss_mine.backward(retain_graph=True)
            else:
                loss_mine.backward()

            accumulated_count+=1
            if accumulated_count==accumulate_num:
                accumulated_count=0
                optimizer_network.step()
                optimizer_mine.step()
                scheduler_network.step()
                scheduler_mine.step()



            predictiont=prediction.cpu()
            prediction_np=predictiont.detach().numpy()
            batch_y0t=batch_y_t.cpu()
            label_np=batch_y0t.detach().numpy()
            for i in range(len(prediction_np)):
                if np.argmax(prediction_np[i])==np.argmax(label_np[i]):
                    train_acc_sum+=1
                train_count+=1


            if step==int(num_train/batch_size)-100:

                model.eval()
                #train result
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
                    prediction_v,x_start_list,x_0list,x_1list=model(batch_xv)
                    x_0=x_start_list[0]
                    x_00=x_0list[0]
                    x_01=x_1list[0]
                    x_1=x_start_list[1]
                    x_10=x_0list[1]
                    x_11=x_1list[1]

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

                term3_1=calculate_mi(m1_ogp,m1_ogq)
                term3_1_np=term3_1.cpu().detach().numpy()
                term4_1=calculate_mi(m1_igp,m1_igq)
                term4_1_np=term4_1.cpu().detach().numpy()

                term3_2=calculate_mi(m2_ogp,m2_ogq)
                term3_2_np=term3_2.cpu().detach().numpy()
                term4_2=calculate_mi(m2_igp,m2_igq)
                term4_2_np=term4_2.cpu().detach().numpy()



                print(' term3_1: ',term3_1_np,' term4_2: ',term4_1_np,
                        ' term3_2: ',term3_2_np,' term4_2: ',term4_2_np)

                print('')

                with open(current_folder+'/'+model_name+'_stdout'+".txt", "a") as std_out:
                    std_out.write('epoch: '+str(epoch)+' step: '+str(step)+' train loss: '+str(train_loss_sum/train_count)+' train acc: '+str(train_acc_sum/train_count)+' val loss: '+str(val_loss_sum/val_count)+' val acc: '+str(val_acc_sum/val_count)+'\n')
                    std_out.write(' term3_1: '+str(term3_1_np)+' term4_1: '+str(term4_1_np)+
                                ' term3_2: '+str(term3_2_np)+' term4_2: '+str(term4_2_np)+' best val acc: '+str(best_val_acc)+'\n')
                    std_out.write('\n')
                    std_out.close()
                torch.save(model.state_dict(), current_folder+'/'+model_name+'/'+model_name)

            batch_x0=None
            batch_y0=None

def try_model():
    head_list=[2,4,8]
    mlp_ratio=4

    num0=225
    num1=64
    num2=16
    token_num_list=[num0,num1,num2]
    dim0=96
    dim1=192
    dim2=384
    token_dim_list=[dim0,dim1,dim2]
    depth_list=[7,1,4]

    change_layer_list=[6,0,None]

    inter_layer_num_=[(num0,num1,32),(num1,num2,10)]
    inter_layer_dim_=[(dim0,dim1),(dim1,dim2)]

    model = sufvit(
        head_list=head_list,
        token_num_list=token_num_list,
        token_dim_list=token_dim_list,
        depth_list=depth_list,
        change_layer_list=change_layer_list,
        inter_layer_num=inter_layer_num_,
        inter_layer_dim=inter_layer_dim_,
        img_size=32,
        patch_size=4,
        stride=2,
        base_dims=[48, 48, 48],
        mlp_ratio=mlp_ratio,
        init = 'trunc',
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )


    mine_heads=1
    MINE_OG_list=[]
    for i in range(len(inter_layer_dim_)):
        MINE_OG=MINE(dim1=inter_layer_dim_[i][1],
                dim2=inter_layer_dim_[i][1],
                num1=inter_layer_num_[i][1],
                num2=inter_layer_num_[i][2],
                qkv_bias=False,
                qk_scale=None,
                drop=0,
                attn_drop=0)
        MINE_OG_list.append(MINE_OG)

    MINE_IG_list=[]
    for i in range(len(inter_layer_dim_)):
        MINE_IG=MINE(dim1=inter_layer_dim_[i][0],
                dim2=inter_layer_dim_[i][1],
                num1=inter_layer_num_[i][0]+1,
                num2=inter_layer_num_[i][2],
                qkv_bias=False,
                qk_scale=None,
                drop=0,
                attn_drop=0)
        MINE_IG_list.append(MINE_IG)

    temp_input=torch.rand(9,3,32,32)

    prediction,x_start_list,x_0list,x_1list=model(temp_input)
    x_0=x_start_list[0]
    x_00=x_0list[0]
    x_01=x_1list[0]
    x_1=x_start_list[1]
    x_10=x_0list[1]
    x_11=x_1list[1]
    ord_list=np.arange(prediction.shape[0])
    np.random.shuffle(ord_list)

    prediction1=prediction.clone().detach()[ord_list]

    x1_0=x_0.clone().detach()[ord_list]
    x1_00=x_00.clone().detach()[ord_list]
    x1_01=x_01.clone().detach()[ord_list]

    x1_1=x_1.clone().detach()[ord_list]
    x1_10=x_10.clone().detach()[ord_list]
    x1_11=x_11.clone().detach()[ord_list]




    m1_ogp=MINE_OG_list[0](x_00,x_01)
    m1_ogq=MINE_OG_list[0](x1_00,x_01)

    m1_igp=MINE_IG_list[0](x_0,x_01)
    m1_igq=MINE_IG_list[0](x1_0,x_01)

    m2_ogp=MINE_OG_list[1](x_10,x_11)
    m2_ogq=MINE_OG_list[1](x1_10,x_11)

    m2_igp=MINE_IG_list[1](x_1,x_11)
    m2_igq=MINE_IG_list[1](x1_1,x_11)

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
