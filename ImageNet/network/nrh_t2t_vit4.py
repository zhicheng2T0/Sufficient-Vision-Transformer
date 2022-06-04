'''
Code modified based on https://github.com/yitu-opensource/T2T-ViT, the corresponding research paper
"Token-to-Token ViT: Training Vision Transformer from Scratch on ImageNet" is available at
https://arxiv.org/abs/2101.11986.
'''

import torch
from torch import nn
import math
from network.trunc_norm import trunc_normal_
from .token_performer import Token_performer
import numpy as np


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
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
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


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))

    def forward(self, x):
        x = self.soft_split0(x).transpose(1, 2)

        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.soft_split1(x).transpose(1, 2)

        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.soft_split2(x).transpose(1, 2)

        x = self.project(x)

        return x

class NRH_T2T_ViT(nn.Module):
    def __init__(self, token_num_list,inter_layer_num,token_dim_list,inter_layer_dim,change_layer_list,
                    img_size=224, patch_size=16, in_chans=3, num_classes=1000, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                  attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, pretrain=False, drop_path_rate=0.1, classifier='token', init = ''):


        super().__init__()


        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrain = pretrain

        self.token_num_list=token_num_list
        self.inter_layer_num=inter_layer_num
        self.token_dim_list=token_dim_list
        self.inter_layer_dim=inter_layer_dim
        self.change_layer_list=change_layer_list

        depth=len(self.token_num_list)

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type='performer', in_chans=in_chans, embed_dim=self.token_dim_list[0], token_dim=64)
        num_patches = self.tokens_to_token.num_patches

        num_token=num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_token, self.token_dim_list[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=token_dim_list[i], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.head = nn.Linear(3*self.token_dim_list[-1],num_classes)

        self.inter0=nn.Sequential(nn.Linear(inter_layer_dim[0][0],inter_layer_dim[0][1]),nn.GELU())
        self.inter00=nn.Sequential(norm_layer(inter_layer_num[0][0]),nn.Linear(inter_layer_num[0][0],inter_layer_num[0][1]),nn.GELU())
        self.inter01=nn.Sequential(norm_layer(inter_layer_num[0][0]),nn.Linear(inter_layer_num[0][0],inter_layer_num[0][2]),nn.GELU())

        self.inter1=nn.Sequential(nn.Linear(inter_layer_dim[1][0],inter_layer_dim[1][1]),nn.GELU())
        self.inter10=nn.Sequential(norm_layer(inter_layer_num[1][0]),nn.Linear(inter_layer_num[1][0],inter_layer_num[1][1]),nn.GELU())
        self.inter11=nn.Sequential(norm_layer(inter_layer_num[1][0]),nn.Linear(inter_layer_num[1][0],inter_layer_num[1][2]),nn.GELU())


        if init == 'ortho':
            self.apply(self._init_weights)
        elif init == 'trunc':
            self.apply(self._init_weights_trunc)
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

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)


        count=0
        for blk in self.blocks:
            if count==self.change_layer_list[0]:
                x_ori=x
            if count==self.change_layer_list[1]:
                x=self.inter0(x)
                xt=torch.transpose(x,2,1)
                xt0=self.inter00(xt)
                xt1=self.inter01(xt)
                x0=torch.transpose(xt0,2,1)
                x1=torch.transpose(xt1,2,1)
                x=torch.cat([x0,x1],1)+x
                x_00=x[:,0:self.inter_layer_num[0][1],:]
                x_01=x[:,self.inter_layer_num[0][1]:,:]
                x=x_00
            if count==self.change_layer_list[2]:
                x_temp=x
            if count==self.change_layer_list[3]:
                x=self.inter1(x)
                xt=torch.transpose(x,2,1)
                xt0=self.inter10(xt)
                xt1=self.inter11(xt)
                x0=torch.transpose(xt0,2,1)
                x1=torch.transpose(xt1,2,1)
                x=torch.cat([x0,x1],1)+x
                x_10=x[:,0:self.inter_layer_num[1][1],:]
                x_11=x[:,self.inter_layer_num[1][1]:,:]
                x=x_10
            x = blk(x)
            count+=1

        return x,x_ori,x_00,x_01,x_temp,x_10,x_11

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if not 'head' in name:
                param.requires_grad = False

    def forward(self, x):
        x,x_ori,x_00,x_01,x_temp,x_10,x_11 = self.forward_features(x)
        x=x[:,0:3,:]
        x=torch.flatten(x,1,2)
        x = self.head(x)
        return x,x_ori,x_00,x_01,x_temp,x_10,x_11



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

heads=4
mlp_ratio=2

num0=196
num1=98
num2=49
token_num_list=[num0,
                num0,
                num0,
                num0,

                num1,num1,

                num2,num2,
                num2,num2,
                num2,num2]

dim0=256
dim1=256
dim2=256
token_dim_list=[dim0,
                dim0,
                dim0,
                dim0,

                dim1,dim1,

                dim2,dim2,
                dim2,dim2,
                dim2,dim2]

change_layer_list=[3,4,5,6]

inter_layer_num_=[(num0,num1,int(num0-num1)),(num1,num2,int(num1-num2))]
inter_layer_dim_=[(dim0,dim1),(dim1,dim2)]


def nhr_t2t_vit_tiny(init, norm_layer, **kwargs):
    model = NRH_T2T_ViT(token_num_list=token_num_list,
                inter_layer_num=inter_layer_num_,
                token_dim_list=token_dim_list,
                inter_layer_dim=inter_layer_dim_,
                change_layer_list=change_layer_list,
                patch_size=16,
                depth=len(token_num_list),
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                init=init,
                norm_layer=norm_layer,
                **kwargs)
    return model

def mine_og0(**kwargs):
    i=0
    model=MINE(dim1=inter_layer_dim_[i][1],
            dim2=inter_layer_dim_[i][1],
            num1=inter_layer_num_[i][1],
            num2=inter_layer_num_[i][2],
            qkv_bias=False,
            qk_scale=None,
            drop=0,
            attn_drop=0)
    return model

def mine_og1(**kwargs):
    i=1
    model=MINE(dim1=inter_layer_dim_[i][1],
            dim2=inter_layer_dim_[i][1],
            num1=inter_layer_num_[i][1],
            num2=inter_layer_num_[i][2],
            qkv_bias=False,
            qk_scale=None,
            drop=0,
            attn_drop=0)
    return model

def mine_ig0(**kwargs):
    i=0
    model=MINE(dim1=inter_layer_dim_[i][0],
                dim2=inter_layer_dim_[i][1],
                num1=inter_layer_num_[i][0],
                num2=inter_layer_num_[i][2],
                qkv_bias=False,
                qk_scale=None,
                drop=0,
                attn_drop=0)
    return model

def mine_ig1(**kwargs):
    i=1
    model=MINE(dim1=inter_layer_dim_[i][0],
                dim2=inter_layer_dim_[i][1],
                num1=inter_layer_num_[i][0],
                num2=inter_layer_num_[i][2],
                qkv_bias=False,
                qk_scale=None,
                drop=0,
                attn_drop=0)
    return model

