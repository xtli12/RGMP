import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from collections import OrderedDict
import torch.nn.functional as F


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):  # swin block中的残差连接
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement) # displacement表示滑动的距离，负号表示向上或者向左
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
            #根据window_size计算相对位置，由于窗口的大小是以中心点为基准向左和向右扩展的，因此最远的相对位置是距离中心点window_size-1个位置。
            #因此，相对位置嵌入矩阵中需要包含从-window_size+1到window_size-1的所有相对位置，即2 * window_size - 1
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
        #将计算得到的注意力头矩阵dots与相对位置嵌入self.pos_embedding相加,得到每个注意力头对应的相对位置偏移量的权重,使模型更好地处理序列中位置关系复杂的任务
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
            #self.relative_indices[:, :, 0]和self.relative_indices[:, :, 1]分别表示相对位置的行坐标和列坐标。
            #这样的索引方式相当于将self.relative_indices中的每个坐标(i, j)作为索引，在self.pos_embedding中获取对应的嵌入向量。
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        ###
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                ### W-MSA
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                ### SW-MSA
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)  # 按通道数来对其进行拼接，dim=1对应（， ， ， ）第二个通道数


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


# class _Transition(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features):
#         super(_Transition, self).__init__()
#         self.add_module('norm', nn.BatchNorm2d(num_input_features))
#         self.add_module('relu', nn.ReLU(inplace=True))
#         self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
#                                           kernel_size=1, stride=1, bias=False))
#         self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# AWCA模块
class AWCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 全连接层
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()  # 输入矩阵参数
        input_x = x  # inputx为叉乘对象
        input_x = input_x.view(b, c, h * w).unsqueeze(1)  # 用unsqueeze扩充一个行维，view调整矩阵维度为c*h*w
        mask = self.conv(x).view(b, 1, h * w)  # 经过卷积层后为1*1*h*w，用view调整为1*1*(h*w)
        mask = self.softmax(mask).unsqueeze(-1)  # 然后对Y进行正规化后扩充一个列维
        y = torch.matmul(input_x, mask).view(b, c)  # 图中的叉乘输出变换为b*c的张量
        y = self.fc(y).view(b, c, 1, 1)  # 对Z进行一系列处理包括两个全连接层、relu和sigmoid
        return x * y.expand_as(x)  # 把输出扩张成原来的大小后点乘原输入x

'''
非本地二阶上下文模块
'''

class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(NONLocalBlock2D, self).__init__()
        # 数组维数读取
        self.in_channels = in_channels # 读入通道数
        self.inter_channels = self.in_channels // reduction # 通道缩放因子原文图中为r大小是8
        conv_nd = nn.Conv2d # 调用二维卷积
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d # 调用归一化处理
        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.constant_(self.W.weight, 0) # 用0填充向量
        # 二维卷积层
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)

    def count_cov_second(self, input):
        x = input
        batchSize, dim, M = x.data.shape
        x_mean_band = x.mean(2).view(batchSize, dim, 1).expand(batchSize, dim, M) # 对第三维度求均值，然后填充成M维
        y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M # 减去第三维度的均值乘以经过变换的x再除以m
        return y

    # 定义网络结构
    def forward(self, x):
        batch_size = x.size(0) # 读取数据并进行维度变换
        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # 1*1*c/r 卷积层,view相当于将四维张量转换成三维张量
        g_x = g_x.permute(0, 2, 1) # 维度转换
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # 卷积层
        theta_x = theta_x.permute(0, 2, 1) # 维度转换
        f = self.count_cov_second(theta_x) # 求协方差矩阵
        f_div_C = F.softmax(f, dim=-1) # softmax归一化
        y = torch.matmul(f_div_C, g_x) # tensor矩阵乘法 即图片里的乘号
        y = y.permute(0, 2, 1).contiguous() # 重组、转置
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y) # 1*1*C卷积层
        z = W_y + x
        return z

'''
downsample环节，卷积核为自定义，目的是缩小图像大小
'''

def pixel_unshuffle(input, downscale_factor):
    c = input.shape[1]
    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor], device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

'''
PSNL模块
'''

class PSNL(nn.Module):
    def __init__(self, channels):
        super(PSNL, self).__init__()
        # nonlocal module 调用NON
        self.non_local = NONLocalBlock2D(channels)

    def forward(self,x):
        batch_size, C, H, W = x.shape # 将输入按照w和h维度分割为大小相同的4块
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)
        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]
        nonlocal_lu = self.non_local(feat_sub_lu) # 实际调用non
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd
        return nonlocal_feat


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=14, head_dim=32, window_size=20,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True,
                 growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule4(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head0 = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):  # 读出索引值和值
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            """下面一行加了i*96"""
            num_features = num_features + num_layers * growth_rate + 96 * 2**i
            # if i != len(block_config) - 1:
            #     trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            #     self.features.add_module('transition%d' % (i + 1), trans)
            #     num_features = num_features // 2
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier0 = nn.Linear(37632, 768)
        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.data)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

        self.awca1 = AWCA(256)
        self.awca2 = AWCA(736)
        self.awca3 = AWCA(1696)
        self.awca4 = AWCA(2592)
        self.ps = PSNL(2592)
        self.cf = nn.Conv2d(4128, 768, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(768)
        self.bn2 = nn.BatchNorm2d(291)
        self.bn3 = nn.BatchNorm2d(675)
        self.bn4 = nn.BatchNorm2d(768)
        self.Re = nn.ReLU(inplace=True)
        self.p0 = nn.MaxPool2d(kernel_size=4, stride=4, padding=1)
        self.p1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.ca = nn.Conv2d(2592, 1536, kernel_size=1, stride=1, bias=False)
        self.cb = nn.Conv2d(291, 192, kernel_size=1, stride=1, bias=False)
        self.cc = nn.Conv2d(675, 384, kernel_size=1, stride=1, bias=False)
        self.cd = nn.Conv2d(1443, 768, kernel_size=1, stride=1, bias=False)
        self.yy = nn.Conv2d(2592, 384, kernel_size=1, stride=1, bias=False)
        self.sapien = nn.Conv2d(4128, 4096, kernel_size=1, stride=1, bias=False)
    def forward(self, img):
        features = self.features.conv0(img)
        features = self.features.norm0(features)
        features = self.features.relu0(features)
        features = self.features.pool0(features)
        features = self.features.denseblock1(features)
        features = self.awca1(features)                 # Dense1 输出
        x = self.stage1(img)                            # swin stage1输出
        a = torch.cat([x, features], 1)                 # 第一个concat
        features = self.features.denseblock2(a)
        features = self.awca2(features)
        features = self.p1(features)                    # Dense2 输出
        x = self.stage2(x)                              # swin stage2输出
        b = torch.cat([x, features], 1)                 # 第二个concat
        features = self.features.denseblock3(b)
        features = self.awca3(features)                 # Dense3 输出
        features = self.p1(features)                    # 经过pooling层
        x = self.stage3(x)
        c = torch.cat([x, features], 1)
        features = self.features.denseblock4(c)         # Dense4 输出
        z = features
        y = self.yy(features)
        x = torch.cat([x, y], 1)                        # c通道 x:前384 y：后384
        x = self.stage4(x)                              # 支路1输出
        features = self.awca4(features)
        features = self.ps(features)
        features = self.ca(features)
        features = self.p1(features)                    # 支路2输出
        out1 = x.mul(features)
        # out = pixel_unshuffle(out1, 2)
        # out = self.cf(out)
        z = self.p1(z)
        out = torch.cat([out1, z], 1)
        out_sapien = self.sapien(out)

        out = self.cf(out)
        # out = self.bn1(out)
        out = self.Re(out)
        # b, c, h, w = out.size()
        # out = out.view(b,c*h*w)
        # out = self.classifier0(out)
        out = out.mean(dim=[2, 3])
        out = self.mlp_head0(out)
        # out = F.view(features.size(0), -1)
        # out = self.classifier0(out)

        # x = x.mean(dim=[2, 3]) # 去除第三四维
        # out1 = self.mlp_head(x)
        # p = out2 + out1
        return out_sapien


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

class StageModule4(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        ###
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock4(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                ### W-MSA
                SwinBlock4(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                ### SW-MSA
            ]))

    def forward(self, x):
        x1 = x[:,:384,:,:]
        x2 = x[:,384:769, :, :]
        x1 = self.patch_partition(x1)
        x2 = self.patch_partition(x2)
        x = torch.cat([x1, x2], 3)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)

class SwinBlock4(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual4(PreNorm4(dim, WindowAttention4(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual4(PreNorm4(dim, FeedForward4(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class Residual4(nn.Module):  # swin block中的残差连接
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm4(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm2d(1536)
        self.fn = fn

    def forward(self, x, **kwargs):
        # x = x.permute(0, 3, 1, 2)
        # x = self.norm(x)
        # x = x.permute(0, 2, 3, 1)
        return self.fn(self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1), **kwargs)

class FeedForward4(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim*2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim*2),
        )

    def forward(self, x):
        return self.net(x)

class WindowAttention4(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, z):
        x = z[:, :, :, :768]
        y = z[:, :, :,768:]
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads
        # b_y, n_h_y, n_w_y, __y, h_y = *y.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        QKV = self.to_qkv(y).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        Q, K, V = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), QKV)
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, K) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, V)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        out = torch.cat([out, y], 3)
        return out

if __name__ == '__main__':
    batch_size = 2

    x = torch.rand(batch_size,6, 1024)
    x = torch.reshape(x, (batch_size, 6, 32, 32))
    C1 = nn.Conv2d(6, 3, kernel_size=1, stride=1, bias=False)
    x = C1(x)
    GSNet = SwinTransformer(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), window_size=1)
    # model = MaskedAutoencoderViT(
    #     patch_size=16, embed_dim=768, depth=12, num_heads=12,
    #     decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #     mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    # resume_file = '/data/home-gxu/lxt21/new_mae/Results/Steel_mae_pre/mae_finetuned_vit_base.pth' # 预加载历史训练模型
    # if resume_file:
    #     if os.path.isfile(resume_file):
    #         print("=> loading checkpoint '{}'".format(resume_file))
    #         checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda(0))
    #         # start_epoch = checkpoint['epoch']
    #         # iteration = checkpoint['iter']
    #         model.load_state_dict(checkpoint['model'], strict = False)
    #         # optimizer.load_state_dict(checkpoint['optimizer'])
    # encoder = model.forward_encoder
    # out = encoder(x,mask_ratio=0.75).size()

    out = GSNet(x)
    out = torch.reshape(out, (batch_size, 1, 64, 64))
    C2 = nn.Conv2d(1, 128, kernel_size=1, stride=1, bias=False)
    out = C2(out)
    out = torch.reshape(out, (batch_size, 512, 1024))


    print(out)
