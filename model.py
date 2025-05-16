""" =================================================

# @Time: 2024/9/2 11:24

# @Author: Gringer

# @File: model.py

# @Software: PyCharm

ELA / CA / CBAM / ECA

================================================== """
import math
from torchvision.models.efficientnet import MBConv
import torch
import torch.nn as nn

'''ELA'''

# 原始
# class MultiScaleELA(nn.Module):
#     def __init__(self, channel, kernel_sizes=[3, 7, 11], reduction_ratio=16):
#         super(MultiScaleELA, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.Conv1d(channel, channel, kernel_size=k, padding=k // 2, groups=channel, bias=False)
#             for k in kernel_sizes
#         ])
#         self.gn = nn.GroupNorm(16, channel)
#         self.reduction_ratio = reduction_ratio
#
#         # 1x1卷积融合层
#         self.conv1x1 = nn.Conv2d(len(kernel_sizes) * channel, channel, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#         # 通道注意力机制（SE模块）
#         self.fc1 = nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1, bias=False)
#         self.fc2 = nn.Conv2d(channel // reduction_ratio, channel, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # 对高度方向进行多尺度卷积
#         x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
#         multi_scale_h = [self.gn(conv(x_h)).view(b, c, h, 1) for conv in self.convs]
#         x_h = torch.cat(multi_scale_h, dim=1)
#
#         # 对宽度方向进行多尺度卷积
#         x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
#         multi_scale_w = [self.gn(conv(x_w)).view(b, c, 1, w) for conv in self.convs]
#         x_w = torch.cat(multi_scale_w, dim=1)
#
#         # 融合多尺度特征并应用1x1卷积
#         x_h = self.conv1x1(x_h)
#         x_w = self.conv1x1(x_w)
#
#         # 融合后的特征图
#         x_combined = x_h * x_w
#
#         # 通道注意力机制
#         x_s = torch.mean(x_combined, dim=[2, 3], keepdim=True)
#         x_s = self.fc1(x_s)
#         x_s = self.relu(x_s)
#         x_s = self.fc2(x_s)
#         x_s = self.sigmoid(x_s)
#
#         # 将通道注意力应用到空间特征图上
#         x_combined = x_combined * x_s
#
#         # 将最终结果应用到输入特征图上
#         return x * x_combined

import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# class MultiScaleELA(nn.Module):
#     def __init__(self, channel, kernel_sizes=[3, 7, 11], reduction_ratio=4):
#         super(MultiScaleELA, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.Conv1d(channel, channel, kernel_size=k, padding=k // 2, groups=channel, bias=False)
#             for k in kernel_sizes
#         ])
#
#         # 动态调整组数的组归一化
#         self.gn = nn.GroupNorm(min(16, channel), channel)
#
#         # BatchNorm 取代了组归一化
#         self.bn = nn.BatchNorm1d(channel)
#
#         self.reduction_ratio = reduction_ratio
#
#         # 1x1卷积融合层，接收多通道特征拼接
#         self.conv1x1 = nn.Conv2d(len(kernel_sizes) * channel, channel, kernel_size=1, bias=False)
#
#         # Swish 代替 ReLU
#         self.swish = Swish()
#
#         # 通道注意力机制（SE模块）
#         self.fc1 = nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1, bias=False)
#         self.fc2 = nn.Conv2d(channel // reduction_ratio, channel, kernel_size=1, bias=False)
#
#         # 可学习的加权参数
#         self.weight_h = nn.Parameter(torch.ones(1))  # 对于高度特征的权重
#         self.weight_w = nn.Parameter(torch.ones(1))  # 对于宽度特征的权重
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # 对高度方向进行多尺度卷积
#         x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
#         multi_scale_h = [self.bn(conv(x_h)).view(b, c, h, 1) for conv in self.convs]
#         x_h = torch.cat(multi_scale_h, dim=1)
#
#         # 对宽度方向进行多尺度卷积
#         x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
#         multi_scale_w = [self.bn(conv(x_w)).view(b, c, 1, w) for conv in self.convs]
#         x_w = torch.cat(multi_scale_w, dim=1)
#
#         # 融合多尺度特征并应用1x1卷积
#         x_h = self.conv1x1(x_h)  # shape: [2, 64, 32, 1]
#         x_w = self.conv1x1(x_w)  # shape: [2, 64, 1, 32]
#
#         # 分别计算 x_h 和 x_w 的注意力
#         # 对 x_h 的通道注意力
#         x_h_s = torch.mean(x_h, dim=[2, 3], keepdim=True)
#         x_h_s = self.fc1(x_h_s)
#         x_h_s = self.swish(x_h_s)
#         x_h_s = self.fc2(x_h_s)
#         x_h_s = torch.sigmoid(x_h_s)
#
#         # 对 x_w 的通道注意力
#         x_w_s = torch.mean(x_w, dim=[2, 3], keepdim=True)
#         x_w_s = self.fc1(x_w_s)
#         x_w_s = self.swish(x_w_s)
#         x_w_s = self.fc2(x_w_s)
#         x_w_s = torch.sigmoid(x_w_s)
#
#         # 应用注意力并广播到原始尺寸
#         x_h = x_h * x_h_s  # shape: [2, 64, 32, 1]
#         x_w = x_w * x_w_s  # shape: [2, 64, 1, 32]
#
#         x_h = x_h.expand(-1, -1, -1, w)  # 广播到 [2, 64, 32, 32]
#         x_w = x_w.expand(-1, -1, h, -1)  # 广播到 [2, 64, 32, 32]
#
#         # 合并注意力结果
#         x_combined = self.weight_h * x_h + self.weight_w * x_w  # shape: [2, 64, 32, 32]
#
#         # 将最终结果应用到输入特征图上
#         return x * x_combined  # 元素级乘法，shape: [2, 64, 32, 32]



class MultiScaleELA(nn.Module):
    def __init__(self, channel, kernel_sizes=[3, 7, 11], reduction_ratio=16):
        super(MultiScaleELA, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channel, channel, kernel_size=k, padding=k // 2, groups=channel, bias=False)
            for k in kernel_sizes
        ])

        # 动态调整组数的组归一化
        self.gn = nn.GroupNorm(min(8, channel), channel)

        # BatchNorm 取代了组归一化
        self.bn = nn.BatchNorm1d(channel)

        self.reduction_ratio = reduction_ratio

        # 1x1卷积融合层，接收多通道特征拼接
        self.conv1x1 = nn.Conv2d(len(kernel_sizes) * channel, channel, kernel_size=1, bias=False)

        # Swish 代替 ReLU
        self.swish = Swish()

        # 通道注意力机制（SE模块）
        self.fc1 = nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction_ratio, channel, kernel_size=1, bias=False)

        # 可学习的加权参数
        self.weight_h = nn.Parameter(torch.ones(1))  # 对于高度特征的权重
        self.weight_w = nn.Parameter(torch.ones(1))  # 对于宽度特征的权重

    def forward(self, x):
        b, c, h, w = x.size()

        # 对高度方向进行多尺度卷积
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        multi_scale_h = [self.bn(conv(x_h)).view(b, c, h, 1) for conv in self.convs]
        x_h = torch.cat(multi_scale_h, dim=1)

        # 对宽度方向进行多尺度卷积
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        multi_scale_w = [self.bn(conv(x_w)).view(b, c, 1, w) for conv in self.convs]
        x_w = torch.cat(multi_scale_w, dim=1)

        # 融合多尺度特征并应用1x1卷积
        x_h = self.conv1x1(x_h)
        x_w = self.conv1x1(x_w)

        # 使用拼接（concatenation）代替加法
        x_combined = torch.cat([x_h, x_w], dim=1)

        # 通道注意力机制
        x_s = torch.mean(x_combined, dim=[2, 3], keepdim=True)
        x_s = self.fc1(x_s)
        x_s = self.swish(x_s)
        x_s = self.fc2(x_s)
        x_s = torch.sigmoid(x_s)

        # 将通道注意力应用到空间特征图上
        x_combined = x_combined * x_s

        # 将最终结果应用到输入特征图上
        return x * x_combined

# 定义包含 ELA 的 MBConv 模块

class MBConvWithMultiScaleELA(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3, reduction_ratio=16):
        super(MBConvWithMultiScaleELA, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.ela = MultiScaleELA(out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        if self.use_residual:
            return x + self.ela(self.block(x))
        else:
            return self.ela(self.block(x))





# 替换 EfficientNet-B0 中的 MBConv 模块


# def replace_mbconv_with_mse(model):
#     for i, block in enumerate(model.features):
#         if isinstance(block, nn.Sequential):  # 如果是一个 Sequential
#             for j, sub_block in enumerate(block):
#                 if isinstance(sub_block, MBConv):
#                     # 解析 MBConv 结构来获取需要的参数
#                     conv_layers = sub_block.block
#                     in_ch = conv_layers[0][0].in_channels  # Conv2dNormActivation 第一个卷积的输入通道数
#                     out_ch = conv_layers[-1][1].num_features  # Conv2dNormActivation 最后一个卷积的输出通道数
#                     stride = conv_layers[0][0].stride  # 获取步幅
#                     expand_ratio = sub_block.block[0][0].out_channels // in_ch  # 扩展比率
#
#                     # 用自定义的 MBConvWithMultiScaleELA 替换
#                     model.features[i][j] = MBConvWithMultiScaleELA(
#                         in_channels=in_ch,
#                         out_channels=out_ch,
#                         expand_ratio=expand_ratio/2,
#                         stride=stride[0] if isinstance(stride, (list, tuple)) else stride
#                     )
#         elif isinstance(block, MBConv):  # 如果是单个 MBConv
#             conv_layers = block.block
#             in_ch = conv_layers[0][0].in_channels
#             out_ch = conv_layers[-1][1].num_features
#             stride = conv_layers[0][0].stride
#             expand_ratio = block.block[0][0].out_channels // in_ch
#
#             # 用自定义的 MBConvWithMultiScaleELA 替换
#             model.features[i] = MBConvWithMultiScaleELA(
#                 in_channels=in_ch,
#                 out_channels=out_ch,
#                 expand_ratio=expand_ratio,
#                 stride=stride[0] if isinstance(stride, (list, tuple)) else stride
#             )
#



# def replace_mbconv_with_mse(model):
#     for i, block in enumerate(model.features):
#         if isinstance(block, nn.Sequential):  # 如果是一个 Sequential
#             for j, sub_block in enumerate(block):
#                 if isinstance(sub_block, MBConv):
#                     # 解析 MBConv 结构来获取需要的参数
#                     conv_layers = sub_block.block
#                     in_ch = conv_layers[0][0].in_channels  # Conv2dNormActivation 第一个卷积的输入通道数
#                     out_ch = conv_layers[-1][1].num_features  # Conv2dNormActivation 最后一个卷积的输出通道数
#                     stride = conv_layers[0][0].stride  # 获取步幅
#                     expand_ratio = sub_block.block[0][0].out_channels // in_ch  # 扩展比率
#
#                     # 用自定义的 MBConvWithMultiScaleELA 替换
#                     # 1
#                     model.features[i][j] = MBConvWithMultiScaleELA(
#                         in_channels=in_ch,
#                         out_channels=out_ch,
#                         expand_ratio=expand_ratio,
#                         stride=stride[0] if isinstance(stride, (list, tuple)) else stride
#                     )
#
#
#         elif isinstance(block, MBConv):  # 如果是单个 MBConv
#             conv_layers = block.block
#             in_ch = conv_layers[0][0].in_channels
#             out_ch = conv_layers[-1][1].num_features
#             stride = conv_layers[0][0].stride
#             expand_ratio = block.block[0][0].out_channels // in_ch
#
#             # 用自定义的 MBConvWithMultiScaleELA 替换
#             model.features[i] = MBConvWithMultiScaleELA(
#                 in_channels=in_ch,
#                 out_channels=out_ch,
#                 expand_ratio=expand_ratio,
#                 stride=stride[0] if isinstance(stride, (list, tuple)) else stride
#             )



import torch.nn as nn

# def replace_mbconv_with_mse(model):
#     for i, block in enumerate(model.features):
#         # 如果是一个 Sequential
#         if isinstance(block, nn.Sequential):
#             orig_modules = list(block)           # 先拆成 list
#             new_modules = []
#             for j, m in enumerate(orig_modules):
#                 if isinstance(m, MBConv):
#                     # 1) 用 MBConvWithMultiScaleELA 替换
#                     # （这部分参数的提取逻辑和你原来的一样）
#                     conv_layers = m.block
#                     in_ch = conv_layers[0][0].in_channels
#                     out_ch = conv_layers[-1][1].num_features
#                     stride = conv_layers[0][0].stride
#                     expand_ratio = conv_layers[0][0].out_channels // in_ch
#
#                     new_modules.append(m)
#
#                     # 2) 在后面插入你想加的那一层，示例用 ExtraLayer
#                     #    把 ExtraLayer 换成你自己的模块，比如 nn.BatchNorm2d(out_ch) 等
#                     extra = MultiDimELA(out_ch)
#                     new_modules.append(extra)
#
#                 else:
#                     # 普通模块直接保留
#                     new_modules.append(m)
#
#             # 3) 重新构造回 nn.Sequential
#             model.features[i] = nn.Sequential(*new_modules)
#
#         # 如果 model.features[i] 本身就是一个 MBConv
#         elif isinstance(block, MBConv):
#             # 同理：先构造替换层 + 插入层，再用 Sequential 包起来
#             conv_layers = block.block
#             in_ch = conv_layers[0][0].in_channels
#             out_ch = conv_layers[-1][1].num_features
#             stride = conv_layers[0][0].stride
#             expand_ratio = conv_layers[0][0].out_channels // in_ch
#             extra = MultiDimELA(out_ch)
#             model.features[i] = nn.Sequential(m,extra)
#
#     return model


def replace_mbconv_with_mse(model):
    for i, block in enumerate(model.features):
        if isinstance(block, nn.Sequential):
            for j, sub_block in enumerate(block):
                if isinstance(sub_block, MBConv):
                    conv_layers = list(sub_block.block)
                    out_ch = conv_layers[-1][0].out_channels  # block[3][0] 是最后的 Conv2d 输出通道

                    # 替换 block[2]：原本是 SqueezeExcitation
                    conv_layers[2] = MultiDimELA(out_ch)

                    # 重组回去
                    sub_block.block = nn.Sequential(*conv_layers)
        elif isinstance(block, MBConv):
            conv_layers = list(block.block)
            out_ch = conv_layers[3][0].out_channels
            conv_layers[2] = MultiDimELA(out_ch)
            block.block = nn.Sequential(*conv_layers)

    return model


def replace_mbconv_with_ela(model):
    for i, block in enumerate(model.features):
        # 如果是一个 Sequential
        if isinstance(block, nn.Sequential):
            orig_modules = list(block)           # 先拆成 list
            new_modules = []
            for j, m in enumerate(orig_modules):
                if isinstance(m, MBConv):
                    # 1) 用 MBConvWithMultiScaleELA 替换
                    # （这部分参数的提取逻辑和你原来的一样）
                    conv_layers = m.block
                    in_ch = conv_layers[0][0].in_channels
                    out_ch = conv_layers[-1][1].num_features
                    stride = conv_layers[0][0].stride
                    expand_ratio = conv_layers[0][0].out_channels // in_ch

                    new_modules.append(m)

                    # 2) 在后面插入你想加的那一层，示例用 ExtraLayer
                    #    把 ExtraLayer 换成你自己的模块，比如 nn.BatchNorm2d(out_ch) 等
                    extra = SingleScaleELA(out_ch)
                    new_modules.append(extra)

                else:
                    # 普通模块直接保留
                    new_modules.append(m)

            # 3) 重新构造回 nn.Sequential
            model.features[i] = nn.Sequential(*new_modules)

        # 如果 model.features[i] 本身就是一个 MBConv
        elif isinstance(block, MBConv):
            # 同理：先构造替换层 + 插入层，再用 Sequential 包起来
            conv_layers = block.block
            in_ch = conv_layers[0][0].in_channels
            out_ch = conv_layers[-1][1].num_features
            stride = conv_layers[0][0].stride
            expand_ratio = conv_layers[0][0].out_channels // in_ch
            extra = SingleScaleELA(out_ch)
            model.features[i] = nn.Sequential(m,extra)

    return model


'''CA'''


class CoordinateAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CoordinateAttention, self).__init__()

        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # b,c,h,w
        _, _, h, w = x.size()
        # (b, c, h, w) --> (b, c, h, 1)  --> (b, c, 1, h)
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # (b, c, h, w) --> (b, c, 1, w)
        x_w = torch.mean(x, dim=2, keepdim=True)
        # (b, c, 1, w) cat (b, c, 1, h) --->  (b, c, 1, h+w)
        # (b, c, 1, h+w) ---> (b, c/r, 1, h+w)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        # (b, c/r, 1, h+w) ---> (b, c/r, 1, h)  、 (b, c/r, 1, w)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        # (b, c/r, 1, h) ---> (b, c, h, 1)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # (b, c/r, 1, w) ---> (b, c, 1, w)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        # s_h往宽方向进行扩展， s_w往高方向进行扩展
        out = (s_h.expand_as(x) * s_w.expand_as(x)) * x

        return out


# 定义包含 CA 的 MBConv 模块
class MBConvWithCA(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3, reduction_ratio=16):
        super(MBConvWithCA, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.ca = CoordinateAttention(out_channels, reduction_ratio)

    def forward(self, x):
        if self.use_residual:
            return x + self.ca(self.block(x))
        else:
            return self.ca(self.block(x))


# 替换 EfficientNet-B0 中的 MBConv 模块
def replace_mbconv_with_ca(model):
    for idx, layer in enumerate(model.features):
        if isinstance(layer, nn.Sequential) and len(layer) == 6:  # EfficientNet MBConv 是一个6层的序列
            in_channels = layer[0].in_channels
            out_channels = layer[5].out_channels
            expand_ratio = layer[0].out_channels // in_channels
            stride = layer[1].stride[0]
            model.features[idx] = MBConvWithCA(in_channels, out_channels, expand_ratio, stride)
    return model


'''CBAM'''


# 通道注意力
class channel_attention(nn.Module):
    def __init__(self, channel, ration=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ration, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view([b, c])
        max_pool = self.max_pool(x).view([b, c])

        avg_fc = self.fc(avg_pool)
        max_fc = self.fc(max_pool)

        out = self.sigmoid(max_fc + avg_fc).view([b, c, 1, 1])
        return x * out


# 空间注意力
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 通道的最大池化
        max_pool = torch.max(x, dim=1, keepdim=True).values
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool_out)
        out = self.sigmoid(conv)

        return out * x


# 将通道注意力和空间注意力进行融合
class CBAM(nn.Module):
    def __init__(self, channel, ration=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(channel, ration)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)

        return out


# 定义包含 CBAM 的 MBConv 模块
class MBConvWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3, reduction_ratio=16):
        super(MBConvWithCBAM, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.cbam = CBAM(out_channels, reduction_ratio)

    def forward(self, x):
        if self.use_residual:
            return x + self.cbam(self.block(x))
        else:
            return self.cbam(self.block(x))


# 替换 EfficientNet-B0 中的 MBConv 模块
def replace_mbconv_with_cbam(model):
    for idx, layer in enumerate(model.features):
        if isinstance(layer, nn.Sequential) and len(layer) == 6:  # EfficientNet MBConv 是一个6层的序列
            in_channels = layer[0].in_channels
            out_channels = layer[5].out_channels
            expand_ratio = layer[0].out_channels // in_channels
            stride = layer[1].stride[0]
            model.features[idx] = MBConvWithCBAM(in_channels, out_channels, expand_ratio, stride)
    return model


'''ECA'''


class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 变成序列的形式
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


# 定义包含 ECA 的 MBConv 模块
class MBConvWithECA(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size=3, reduction_ratio=16):
        super(MBConvWithECA, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.eca = ECA(out_channels, reduction_ratio)

    def forward(self, x):
        if self.use_residual:
            return x + self.eca(self.block(x))
        else:
            return self.eca(self.block(x))


# 替换 EfficientNet-B0 中的 MBConv 模块
def replace_mbconv_with_eca(model):
    for idx, layer in enumerate(model.features):
        if isinstance(layer, nn.Sequential) and len(layer) == 6:  # EfficientNet MBConv 是一个6层的序列
            in_channels = layer[0].in_channels
            out_channels = layer[5].out_channels
            expand_ratio = layer[0].out_channels // in_channels
            stride = layer[1].stride[0]
            model.features[idx] = MBConvWithECA(in_channels, out_channels, expand_ratio, stride)
    return model



class MultiDimELA(nn.Module):
    def __init__(self, channels, kernel_sizes=(3,7,11), reduction=16):
        """
        channels: 输入/输出的通道数 C
        kernel_sizes: 高/宽 分支中 1D 卷积的多尺度内核大小
        reduction: cSE 分支中通道压缩比
        """
        super().__init__()
        self.channels = channels
        self.ks = kernel_sizes
        # -------- Height branch --------
        self.h_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k//2, groups=channels, bias=False)
            for k in self.ks
        ])
        self.h_gns = nn.ModuleList([
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
            for _ in self.ks
        ])
        # 将 concat 后的 3C -> C
        self.h_reduce = nn.Conv2d(len(self.ks)*channels, channels, kernel_size=1, bias=False)

        # -------- Width branch --------
        self.w_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k//2, groups=channels, bias=False)
            for k in self.ks
        ])
        self.w_gns = nn.ModuleList([
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
            for _ in self.ks
        ])
        self.w_reduce = nn.Conv2d(len(self.ks)*channels, channels, kernel_size=1, bias=False)

        # -------- Channel (cSE) branch --------
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # --- Height branch ---
        # pool width → (B, C, H, 1)
        h = F.avg_pool2d(x, kernel_size=(1, W))      # (B, C, H, 1)
        h = h.squeeze(-1)                            # (B, C, H)
        # 三个尺度的 1D 卷积 + GN
        h_feats = []
        for conv, gn in zip(self.h_convs, self.h_gns):
            y = conv(h)                              # (B, C, H)
            y = gn(y)                                # (B, C, H)
            h_feats.append(y)
        # concat → (B, 3C, H) → unsqueeze → (B, 3C, H, 1)
        h_cat = torch.cat(h_feats, dim=1).unsqueeze(-1)
        # 1×1 conv 降回 C → (B, C, H, 1)
        h_att = self.h_reduce(h_cat)
        h_att = torch.sigmoid(h_att)                # 注意力映射
        h_att = h_att.expand_as(x)                  # expand → (B, C, H, W)

        # --- Width branch ---
        w = F.avg_pool2d(x, kernel_size=(H, 1))      # (B, C, 1, W)
        w = w.squeeze(2)                             # (B, C, W)
        w_feats = []
        for conv, gn in zip(self.w_convs, self.w_gns):
            y = conv(w)                              # (B, C, W)
            y = gn(y)                                # (B, C, W)
            w_feats.append(y)
        w_cat = torch.cat(w_feats, dim=1).unsqueeze(2)  # (B, 3C, 1, W)
        w_att = self.w_reduce(w_cat)                 # (B, C, 1, W)
        w_att = torch.sigmoid(w_att)
        w_att = w_att.expand_as(x)                   # (B, C, H, W)

        # --- Channel (cSE) branch ---
        c = F.adaptive_avg_pool2d(x, output_size=1).view(B, C)  # (B, C)
        c = self.fc1(c)                                  # (B, C//r)
        c = F.relu(c, inplace=True)
        c = self.fc2(c)                                  # (B, C)
        c_att = self.sigmoid(c).view(B, C, 1, 1)         # (B, C, 1, 1)
        c_att = c_att.expand_as(x)                       # (B, C, H, W)

        # --- 融合 ---
        out = x * h_att * w_att * c_att
        return out



class SingleScaleELA(nn.Module):
    def __init__(self, channels, kernel_size=7, reduction=16):
        """
        channels: 输入/输出通道数 C
        kernel_size: Height/Width 分支中 1D 卷积的核大小（单尺度）
        reduction: cSE 分支中通道压缩比
        """
        super().__init__()
        self.channels = channels

        # --- Height branch (single-scale) ---
        self.h_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )
        self.h_gn = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.h_reduce = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # --- Width branch (single-scale) ---
        self.w_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )
        self.w_gn = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.w_reduce = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # --- Channel (cSE) branch ---
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Height branch
        # 1) 对宽度做全局 avgpool → (B, C, H, 1) → squeeze → (B, C, H)
        h = F.avg_pool2d(x, (1, W)).squeeze(-1)
        # 2) depthwise Conv1d + GN → (B, C, H)
        h = self.h_conv(h)
        h = self.h_gn(h)
        # 3) 恢复为 (B, C, H, 1)，再 1×1 conv → (B, C, H, 1)
        h = self.h_reduce(h.unsqueeze(-1))
        # 4) Sigmoid & expand
        h_att = torch.sigmoid(h).expand_as(x)

        # Width branch
        w = F.avg_pool2d(x, (H, 1)).squeeze(2)  # (B, C, W)
        w = self.w_conv(w)
        w = self.w_gn(w)
        w = self.w_reduce(w.unsqueeze(2))  # (B, C, 1, W)
        w_att = torch.sigmoid(w).expand_as(x)

        # Channel branch (cSE)
        c = F.adaptive_avg_pool2d(x, 1).view(B, C)
        c = F.relu(self.fc1(c), inplace=True)
        c = self.fc2(c)
        c_att = self.sigmoid(c).view(B, C, 1, 1).expand_as(x)

        # 融合三路注意力
        return x * h_att * w_att * c_att
