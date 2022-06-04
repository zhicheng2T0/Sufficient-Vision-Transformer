import torch
from torch import nn
import math
from network.trunc_norm import trunc_normal_

from einops import rearrange
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

        self.c_dim_old=int(dim_old/2)
        self.c_dim_new=int(dim_new/2)


        self.channel0_0=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel0_1=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel1_0=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())
        self.channel1_1=torch.nn.Sequential(torch.nn.Linear(num_old,num_new),torch.nn.GELU())

        self.fc = nn.Linear(dim_old, dim_new)

    def forward(self, x, cls_token):

        x=torch.transpose(x,1,2)
        x0=x[:,0:self.c_dim_old,:]
        x1=x[:,self.c_dim_old:,:]


        x0_0=self.channel0_0(x0)
        x0_1=self.channel0_1(x0)
        x1_0=self.channel1_0(x1)
        x1_1=self.channel1_1(x1)

        x=torch.transpose(torch.cat([x0_0,x0_1,x1_0,x1_1],1),1,2)

        if cls_token!=None:
            cls_token = self.fc(cls_token)
            return x, cls_token
        else:
            return x

class suf_vit(nn.Module):
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

    def forward(self, x,allout=True):
        cls_token,x_start_list,x_0_list,x_1_list = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        if allout==True:
            return cls_token,x_start_list,x_0_list,x_1_list
        else:
            return cls_token


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




