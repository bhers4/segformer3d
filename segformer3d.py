import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from loguru import logger


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W, D)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2,
                                     self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2,
                                     self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleMlp(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):
        x = self.fc1(x)
        x = self.dwconv(x, H, W, D)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, D))
        return x


class SegFormer(nn.Module):

    def __init__(self, zoom_size, patch_size=7, in_channels=1, num_classes=1,
                 embedding_dim=[128, 128, 256, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.05, drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[2, 3, 6, 3], sr_ratios=[8, 4, 2, 1], initial_dims=64,
                 decoder_features=64,
                 shallow=False, decoder_dropout=0.3, add_skip=True) -> None:
        super().__init__()
        self.zoom_size = zoom_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dims = embedding_dim
        self.all_heads = num_heads
        self.all_mlp = mlp_ratios
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.depths = depths
        self.sr_ratios = sr_ratios

        self.num_classes = num_classes
        self.depths = depths

        self.initial_dims = initial_dims
        # self.initial_dims = 64
        self.initial_conv = nn.Conv3d(in_channels=in_channels, out_channels=self.initial_dims, stride=2, padding=1, kernel_size=3)
        self.relu = nn.GELU()
        self.inital_norm = nn.LayerNorm((zoom_size // 2, zoom_size // 2, zoom_size // 2))

        self.shallow = shallow
        if self.shallow:
            self.patch_embed1 = OverlapPatchEmbedding(zoom_size=zoom_size // 2, patch_size=3, stride=2, in_channels=self.initial_dims,
                                                embedding_dim=embedding_dim[0], padding=1)
            self.patch_embed2 = OverlapPatchEmbedding(zoom_size=zoom_size // 4, patch_size=3, stride=2, in_channels=embedding_dim[0],
                                                embedding_dim=embedding_dim[1], padding=1)
            self.patch_embed3 = OverlapPatchEmbedding(zoom_size=zoom_size // 8, patch_size=3, stride=1, in_channels=embedding_dim[1],
                                                embedding_dim=embedding_dim[2], padding=1)
            self.patch_embed4 = OverlapPatchEmbedding(zoom_size=zoom_size // 8, patch_size=3, stride=1, in_channels=embedding_dim[2],
                                                embedding_dim=embedding_dim[3], padding=1)
        else:
            self.patch_embed1 = OverlapPatchEmbedding(zoom_size=zoom_size, patch_size=3, stride=2, in_channels=self.initial_dims,
                                                embedding_dim=embedding_dim[0], padding=1)
            self.patch_embed2 = OverlapPatchEmbedding(zoom_size=zoom_size // 4, patch_size=3, stride=2, in_channels=embedding_dim[0],
                                                embedding_dim=embedding_dim[1], padding=1)
            self.patch_embed3 = OverlapPatchEmbedding(zoom_size=zoom_size // 8, patch_size=3, stride=2, in_channels=embedding_dim[1],
                                                embedding_dim=embedding_dim[2], padding=1)
            self.patch_embed4 = OverlapPatchEmbedding(zoom_size=zoom_size // 16, patch_size=3, stride=2, in_channels=embedding_dim[2],
                                                embedding_dim=embedding_dim[3], padding=1)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embedding_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        self.norm1 = norm_layer(embedding_dim[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embedding_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embedding_dim[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embedding_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embedding_dim[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embedding_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embedding_dim[3])
        self.apply(self._init_weights)

        self.decoder = SegFormerDecoderHead(in_channels=embedding_dim,
                                            decoder_features=decoder_features, 
                                            zoom_size=zoom_size,
                                            decoder_dropout=decoder_dropout,
                                            add_skip=add_skip,
                                            initial_dims=self.initial_dims)
        
        # Attention
        self.block1_grad = None
        self.block2_grad = None
        self.block3_grad = None
        self.block4_grad = None


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def activation_hook1(self, grad):
        self.block1_grad = grad
        return
    
    def activation_hook2(self, grad):
        self.block2_grad = grad
        return
    
    def activation_hook3(self, grad):
        self.block3_grad = grad
        return
    
    def activation_hook4(self, grad):
        self.block4_grad = grad
        return

    def forward(self, x):
        B = x.shape[0]
        outs = []

        out = self.initial_conv(x)
        out = self.relu(out)
        out = self.inital_norm(out)
        # stage 1
        x, H, W, D = self.patch_embed1(out)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W, D)
        x = self.norm1(x)
        x = x.reshape(B, H, W, D, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 2
        x, H, W, D = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W, D)
        x = self.norm2(x)
        x = x.reshape(B, H, W, D, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 3
        x, H, W, D = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W, D)
        x = self.norm3(x)
        x = x.reshape(B, H, W, D, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # stage 4
        x, H, W, D = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W, D)
        x = self.norm4(x)
        x = x.reshape(B, H, W, D, -1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)
        # Potential Skip Connection candidate
        outs.append(out)
        seg_map = self.decoder(outs)
        return seg_map


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, zoom_size, patch_size=7, stride=4, in_channels=1,
                 embedding_dim=128, padding=3) -> None:
        super().__init__()

        self.img_size = (zoom_size, zoom_size, zoom_size)
        self.patch_size = (patch_size, patch_size, patch_size)
        self.H = self.img_size[0] // self.patch_size[0]
        self.W = self.img_size[1] // self.patch_size[1]
        self.D = self.img_size[2] // self.patch_size[2]
        # Projection from input channel dim to embedding dim
        self.proj = nn.Conv3d(in_channels, embedding_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=padding)
        self.norm = nn.LayerNorm(embedding_dim)
        # TODO see what happens with/without this
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.proj(x)
        _, _, height, width, depth = out.shape
        # Flatten
        out = out.flatten(2).transpose(1, 2)
        return self.norm(out), height, width, depth


class SegFormerDecoderHead(nn.Module):

    def __init__(self, in_channels=[128, 128, 256, 512], embedding_dim=256,
                 zoom_size=128, decoder_features=64, decoder_dropout=0.3,
                 add_skip=False, initial_dims=64) -> None:
        super().__init__()

        self.linear_c4 = SimpleMlp(input_dim=in_channels[3],
                                   embed_dim=embedding_dim)
        self.linear_c3 = SimpleMlp(input_dim=in_channels[2],
                                   embed_dim=embedding_dim)
        self.linear_c2 = SimpleMlp(input_dim=in_channels[1],
                                   embed_dim=embedding_dim)
        self.linear_c1 = SimpleMlp(input_dim=in_channels[0],
                                   embed_dim=embedding_dim)

        self.linear_fuse = nn.Conv3d(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=3,
            padding=1
        )
        # Norm
        self.norm_quarter = nn.LayerNorm((zoom_size//4, zoom_size // 4,
                                          zoom_size // 4))
        self.norm_half = nn.LayerNorm((zoom_size//2, zoom_size//2, zoom_size//2))
        self.norm_full = nn.LayerNorm((zoom_size, zoom_size, zoom_size))
        self.relu = nn.GELU()
        self.dropout = nn.Dropout3d(p=decoder_dropout)
        self.add_skip = add_skip
        if add_skip:
            self.conv_smoother = nn.Conv3d(embedding_dim + initial_dims,
                                           embedding_dim, kernel_size=3,
                                           padding=1)
        else:
            self.conv_smoother = nn.Conv3d(embedding_dim, embedding_dim,
                                           kernel_size=3, padding=1)
        self.conv_fullsize_smoother = nn.Conv3d(embedding_dim, decoder_features,
                                                kernel_size=3, padding=1)
        self.linear_pred = nn.Conv3d(decoder_features, 1, kernel_size=3,
                                     padding=1)
    
    def forward(self, x):
        c1, c2, c3, c4, initial_conv = x
        c1_shape = c1.shape
        c2_shape = c2.shape
        c3_shape = c3.shape
        c4_shape = c4.shape
        c1 = c1.view(c1.shape[0], c1.shape[1], -1)
        c2 = c2.view(c2.shape[0], c2.shape[1], -1)
        c3 = c3.view(c3.shape[0], c3.shape[1], -1)
        c4 = c4.view(c4.shape[0], c4.shape[1], -1)

        n, _, _ = c4.shape

        _c4 = self.linear_c4(c4).transpose(2, 1)
        _c4 = _c4.view(n, -1, c4_shape[2], c4_shape[3], c4_shape[4])
        c4_scale = c1_shape[2] / c4_shape[2]
        _c4 = F.interpolate(_c4, scale_factor=c4_scale)

        _c3 = self.linear_c3(c3).transpose(2, 1)
        _c3 = _c3.view(n, -1, c3_shape[2], c3_shape[3], c3_shape[4])
        c3_scale = c1_shape[2] / c3_shape[2]
        _c3 = F.interpolate(_c3, scale_factor=c3_scale)

        _c2 = self.linear_c2(c2).transpose(2, 1)
        _c2 = _c2.view(n, -1, c2_shape[2], c2_shape[3], c2_shape[4])
        c2_scale = c1_shape[2] / c2_shape[2]
        _c2 = F.interpolate(_c2, scale_factor=c2_scale)

        _c1 = self.linear_c1(c1).transpose(2, 1)
        _c1 = _c1.view(n, -1, c1_shape[2], c1_shape[3], c1_shape[4])
        # Block, need conv + relu + norm
        stacked_c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        x = self.linear_fuse(stacked_c)
        x = self.relu(x)
        x = self.norm_quarter(x)
        x = self.dropout(x)
        # print("After fusing/stacking all the segformer blocks together: ", x.shape)
        # Block, need conv + relu + norm
        x = F.interpolate(x, scale_factor=2)
        # Concat
        if self.add_skip:
            x = torch.cat([x, initial_conv], dim=1)
        x = self.conv_smoother(x)
        x = self.relu(x)
        x = self.norm_half(x)
        x = self.dropout(x)
        # Block
        x = F.interpolate(x, scale_factor=2)
        # Now at original resolution
        x = self.conv_fullsize_smoother(x)
        x = self.relu(x)
        x = self.norm_full(x)
        # No dropout here
        x = self.linear_pred(x)
        return x