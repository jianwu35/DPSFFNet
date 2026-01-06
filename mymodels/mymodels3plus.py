import numpy as np
import torch
from conv_lstm import ConvLSTM
import torch.fft as fft
from einops import rearrange, repeat
from fft_conv_pytorch import FFTConv2d
from thop import profile
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from typing_extensions import Sequence
from 超声心动图.超声心动图.Echotest.echonet.utils.models.CBAM import CBAM
from 超声心动图.超声心动图.ISIC.mymodels.tezhengronghe import mynet


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.cbam = CBAM(out_channels * len(self.convs))
        # self.ema = EMA(out_channels * len(self.convs))
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        res = self.cbam(res)
        # res = self.ema(res)
        return self.project(res)


class EncoderBottleneck1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, base_width=64):
        super().__init__()  # 初始化父类
        self.downsample = nn.Sequential(  # 下采样层，用于降低特征图的维度
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        width = int(out_channels * (base_width / 64))  # 计算中间通道数
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)  # 第一个卷积层
        self.norm1 = nn.BatchNorm2d(width)  # 第一个批量归一化层
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1,
                               bias=False)  # 第二个卷积层
        self.norm2 = nn.BatchNorm2d(width)  # 第二个批量归一化层
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1)  # 第三个卷积层
        self.norm3 = nn.BatchNorm2d(out_channels)  # 第三个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, x):
        x_down = self.downsample(x)  # 下采样操作
        x = self.conv1(x)  # 第一个卷积操作
        x = self.norm1(x)  # 第一个批量归一化
        x = self.relu(x)  # ReLU激活
        x = self.conv2(x)  # 第二个卷积操作
        x = self.norm2(x)  # 第二个批量归一化
        x = self.relu(x)  # ReLU激活
        x = self.conv3(x)  # 第三个卷积操作
        x = self.norm3(x)  # 第三个批量归一化
        x = x + x_down  # 残差连接
        x = self.relu(x)  # ReLU激活
        return x


class EncoderBottleneck2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, base_width=64):
        super().__init__()  # 初始化父类
        self.downsample = nn.Sequential(  # 下采样层，用于降低特征图的维度
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        width = int(out_channels * (base_width / 64))  # 计算中间通道数
        self.conv1 = FFTConv2d(in_channels, width, kernel_size=1, stride=1, bias=False)  # 第一个卷积层
        self.norm1 = nn.BatchNorm2d(width)  # 第一个批量归一化层
        self.conv2 = FFTConv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1,
                               bias=False)  # 第二个卷积层
        self.norm2 = nn.BatchNorm2d(width)  # 第二个批量归一化层
        self.conv3 = FFTConv2d(width, out_channels, kernel_size=1, stride=1, bias=False)  # 第三个卷积层
        self.norm3 = nn.BatchNorm2d(out_channels)  # 第三个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

    def forward(self, x):
        x_down = self.downsample(x)  # 下采样操作
        x = self.conv1(x)  # 第一个卷积操作

        x = self.norm1(x)  # 第一个批量归一化
        x = self.relu(x)  # ReLU激活
        x = self.conv2(x)  # 第二个卷积操作

        x = self.norm2(x)  # 第二个批量归一化
        x = self.relu(x)  # ReLU激活
        x = self.conv3(x)  # 第三个卷积操作

        x = self.norm3(x)  # 第三个批量归一化
        x = x + x_down  # 残差连接
        x = self.relu(x)  # ReLU激活
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()  # 调用父类构造函数

        self.head_num = head_num  # 多头的数量
        self.dk = (embedding_dim // head_num) ** (1 / 2)  # 缩放因子，用于缩放点积注意力

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)  # 线性层，用于生成查询（Q）、键（K）和值（V）
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)  # 输出线性层

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)  # 通过线性层生成Q、K、V

        query, key, value = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_num))  # 将Q、K、V重塑为多头注意力的格式
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk  # 计算点积注意力的能量

        if mask is not None:  # 如果提供了掩码，则在能量上应用掩码z
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)  # 应用softmax函数，得到注意力权重

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)  # 应用注意力权重到值上

        x = rearrange(x, "b h t d -> b t (h d)")  # 重塑x以准备输出
        x = self.out_attention(x)  # 通过输出线性层
        return x


# 定义MLP模块
class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()  # 调用父类构造函数

        self.mlp_layers = nn.Sequential(  # 定义MLP的层
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),  # GELU激活函数
            nn.Dropout(0.1),  # Dropout层，用于正则化
            nn.Linear(mlp_dim, embedding_dim),  # 线性层
            nn.Dropout(0.1)  # Dropout层
        )

    def forward(self, x):
        x = self.mlp_layers(x)  # 通过MLP层
        return x


# 定义Transformer编码器块
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()  # 调用父类构造函数

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)  # 多头注意力模块
        self.mlp = MLP(embedding_dim, mlp_dim)  # MLP模块

        self.layer_norm1 = nn.LayerNorm(embedding_dim)  # 第一层归一化
        self.layer_norm2 = nn.LayerNorm(embedding_dim)  # 第二层归一化

        self.dropout = nn.Dropout(0.1)  # Dropout层

    def forward(self, x):
        _x = self.multi_head_attention(x)  # 通过多头注意力模块
        _x = self.dropout(_x)  # 应用dropout
        x = x + _x  # 残差连接
        x = self.layer_norm1(x)  # 第一层归一化

        _x = self.mlp(x)  # 通过MLP模块
        x = x + _x  # 残差连接
        x = self.layer_norm2(x)  # 第二层归一化

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()  # 调用父类构造函数

        self.layer_blocks = nn.ModuleList([  # 创建一个模块列表，包含多个编码器块
            TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)
        ])

    def forward(self, x):
        for layer_block in self.layer_blocks:  # 遍历每个编码器块
            x = layer_block(x)  # 通过每个块
        return x


# 定义ViT模型
class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim,
                 classification=True, num_classes=3):
        super().__init__()  # 调用父类构造函数

        self.patch_dim = patch_dim  # 定义patch的维度

        self.classification = classification  # 是否进行分类
        self.num_tokens = (img_dim // patch_dim) ** 2  # 计算tokens的数量

        self.token_dim = in_channels * (patch_dim ** 2)  # 计算每个token的维度
        self.projection = nn.Linear(self.token_dim, embedding_dim)  # 线性层，用于将patches投影到embedding空间
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))  # 可学习的embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))  # 类别token

        self.dropout = nn.Dropout(0.1)  # Dropout层

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)  # Transformer编码器

        if self.classification:  # 如果是分类任务
            self.mlp_head = nn.Linear(embedding_dim, num_classes)  # 分类头

    def forward(self, x):
        img_patches = rearrange(x,  # 将输入图像重塑为patches序列
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        batch_size, tokens, _ = img_patches.shape  # 获取批次大小、tokens数量和通道数

        project = self.projection(img_patches)  # 将patches投影到embedding空间

        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)  # 重复cls_token以匹配批次大小
        patches = torch.cat((token, project), dim=1)  # 将cls_token和投影后的patches拼接

        patches += self.embedding[:tokens + 1, :]  # 将可学习的embedding添加到patches

        x = self.dropout(patches)  # 应用dropout

        x = self.transformer(x)  # 通过Transformer编码器

        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]  # 如果是分类任务，使用cls_token的输出；否则，使用patches的输出

        return x


# 定义解码器中的瓶颈层
class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, concat_channels=None):
        super().__init__()  # 初始化父类
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)  # 上采样层
        self.layer = nn.Sequential(  # 解码器层
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)  # 上采样操作
        if x_concat is not None:  # 如果有额外的特征图进行拼接
            x = torch.cat([x_concat, x], dim=1)  # 在通道维度上拼接
        x = self.layer(x)  # 通过解码器层
        return x


# 定义编码器
class Encoder1(nn.Module):
    def __init__(self, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, img_dim):
        super(Encoder1, self).__init__()  # 初始化父类
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, dilation=1)  # 第一个卷积层
        self.norm1 = nn.BatchNorm2d(out_channels)  # 第一个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.encoder1 = EncoderBottleneck1(out_channels, out_channels * 4, stride=2)  # 第一个编码器瓶颈层
        self.encoder2 = EncoderBottleneck1(out_channels * 4, out_channels * 8, stride=2)  # 第二个编码器瓶颈层
        self.encoder3 = EncoderBottleneck1(out_channels * 8, out_channels * 16, stride=2)  # 第三个编码器瓶颈层
        self.vit_img_dim = img_dim // patch_dim  # ViT的图像维度
        self.vit = ViT(self.vit_img_dim, out_channels * 16, out_channels * 32,  # ViT模型
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        self.conv2 = nn.Conv2d(out_channels * 32, 512, kernel_size=3, stride=1, padding=1)  # 第四个卷积层
        self.norm2 = nn.BatchNorm2d(512)  # 第四个批量归一化层

    def forward(self, x):
        x = self.conv1(x)  # 第一个卷积操作
        x = self.norm1(x)  # 第一个批量归一化
        x1 = self.relu(x)  # ReLU激活
        x2 = self.encoder1(x1)  # 第一个编码器瓶颈层
        x3 = self.encoder2(x2)  # 第二个编码器瓶颈层
        x = self.encoder3(x3)  # 第三个编码器瓶颈层
        x = self.vit(x)  # 通过ViT模型
        x = rearrange(x, "b  (x y) c  -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # 重塑特征图
        x = self.conv2(x)  # 第四个卷积操作
        x = self.norm2(x)  # 第四个批量归一化
        x = self.relu(x)  # ReLU激活
        return x, x1, x2, x3  # 返回多个特征图


class Encoder2(nn.Module):
    def __init__(self, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, img_dim):
        super(Encoder2, self).__init__()  # 初始化父类
        self.conv1 = FFTConv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)  # 第一个卷积层
        self.norm1 = nn.BatchNorm2d(out_channels)  # 第一个批量归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.encoder1 = EncoderBottleneck2(out_channels, out_channels * 4, stride=2)  # 第一个编码器瓶颈层
        self.encoder2 = EncoderBottleneck2(out_channels * 4, out_channels * 8, stride=2)  # 第二个编码器瓶颈层
        self.encoder3 = EncoderBottleneck2(out_channels * 8, out_channels * 16, stride=2)  # 第三个编码器瓶颈层
        self.vit_img_dim = img_dim // patch_dim  # ViT的图像维度
        self.vit = ViT(self.vit_img_dim, out_channels * 16, out_channels * 32,  # ViT模型
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        self.conv2 = nn.Conv2d(out_channels * 32, 512, kernel_size=3, stride=1, padding=1)  # 第四个卷积层
        self.norm2 = nn.BatchNorm2d(512)  # 第四个批量归一化层

    def forward(self, y):
        y = self.conv1(y)  # 第一个卷积操作
        y = self.norm1(y)  # 第一个批量归一化
        y1 = self.relu(y)  # ReLU激活
        y2 = self.encoder1(y1)  # 第一个编码器瓶颈层
        y3 = self.encoder2(y2)  # 第二个编码器瓶颈层
        y = self.encoder3(y3)  # 第三个编码器瓶颈层
        y = self.vit(y)  # 通过ViT模型
        y = rearrange(y, "b (x y) c   -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # 重塑特征图
        y = self.conv2(y)  # 第四个卷积操作
        y = self.norm2(y)  # 第四个批量归一化
        y = self.relu(y)  # ReLU激活
        return y, y1, y2, y3  # 返回多个特征图


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()  # 初始化父类
        self.decoder1 = DecoderBottleneck(536, 256)  # 第一个解码器瓶颈层
        self.decoder2 = DecoderBottleneck(268, 128)  # 第二个解码器瓶颈层
        self.decoder3 = DecoderBottleneck(131, 64)  # 第三个解码器瓶颈层
        self.decoder4 = DecoderBottleneck(64, out_channels)  # 第四个解码器瓶颈层
        self.conv1 = nn.Conv2d(out_channels, class_num, kernel_size=1)  # 最后一个卷积层，用于输出

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)  # 第一个解码器瓶颈层
        x = self.decoder2(x, x2)  # 第二个解码器瓶颈层
        x = self.decoder3(x, x1)  # 第三个解码器瓶颈层
        x = self.decoder4(x)  # 第四个解码器瓶颈层
        x = self.conv1(x)  # 最后一个卷积层
        return x  # 返回解码器的输出


# 定义TransUNet模型
class TransUNet3plus(nn.Module):
    def __init__(self, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num, img_dim):
        super().__init__()  # 初始化父类
        self.encoder1 = Encoder1(in_channels, out_channels,  # 初始化编码器
                                 head_num, mlp_dim, block_num, patch_dim, img_dim)
        #self.encoder2 = Encoder2(in_channels, out_channels,  # 初始化编码器
                                # head_num, mlp_dim, block_num, patch_dim, img_dim)

        #self.aspp0 = ASPP(256, atrous_rates=[1, 4, 8, 12])  # 初始化ASPP模块

        #self.aspp1 = ASPP(out_channels * 2, atrous_rates=[1, 4, 8, 12])  # 初始化ASPP模块
        #self.aspp2 = ASPP(out_channels * 4, atrous_rates=[1, 4, 8, 12])  # 初始化ASPP模块
        #self.aspp3 = ASPP(out_channels * 8, atrous_rates=[1, 4, 8, 12])  # 初始化ASPP模块

        #self.net0 = mynet(128, 16)
        #self.net1 = mynet(out_channels, 1)
        #self.net2 = mynet(out_channels * 2, 2)
        #self.net3 = mynet(out_channels * 4, 4)
        # self.net0 = Mynet(512, 512)
        # self.net1 = Mynet(out_channels, 512)
        # self.net2 = Mynet(out_channels * 2, 512)
        # self.net3 = Mynet(out_channels * 4, 512)
        self.decoder = Decoder(out_channels, class_num)  # 初始化解码器

    def forward(self, x):
        x0, x1, x2, x3 = self.encoder1(x)  # 编码分支
        #y0, y1, y2, y3 = self.encoder2(x)  # 编码分支
        """
        x0 = self.net0(x0, y0)
        x1 = self.net1(x1, y1)
        x2 = self.net2(x2, y2)
        x3 = self.net3(x3, y3)

        x0 = self.aspp0(x0)  # 应用ASPP模块
        x1 = self.aspp1(x1)  # 应用ASPP模块
        x2 = self.aspp2(x2)  # 应用ASPP模块
        x3 = self.aspp3(x3)  # 应用ASPP模块
        
        
        x0 = torch.cat((x0, y0), dim=1)  # 拼接操作
        x1 = torch.cat((x1, y1), dim=1)  # 拼接操作
        x2 = torch.cat((x2, y2), dim=1)  # 拼接操作
        x3 = torch.cat((x3, y3), dim=1)  # 拼接操作
        x0 = self.aspp0(x0)  # 应用ASPP模块
        x1 = self.aspp1(x1)  # 应用ASPP模块
        x2 = self.aspp2(x2)  # 应用ASPP模块
        x3 = self.aspp3(x3)  # 应用ASPP模块
        """
        # lstm_out = self.convlstm(x0)
        # lstm_out = lstm_out[0]
        # x = self.decoder(lstm_out[0][:, -1, :, :], x1, x2, x3)  # 解码分支

        x = self.decoder(x0, x1, x2, x3)  # 解码分支

        return x  # 返回最终输出


def cal_params_flops(model, size):
    input = torch.randn(1, 1, size, size)
    input = input.to('cuda')
    flops, params = profile(model, inputs=(input,))
    Gflops = flops / 1e9
    MACs = flops / 2
    print('Gflops', Gflops)  # 打印计算量
    print('params', params / 1e6)  # 打印参数量
    print('MACs', MACs / 1e9)  # 打印计算量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    img_dim = 256  # 图像的维度，可能是宽度和高度
    head_num = 8  # Transformer 头的数量
    mlp_dim = 128  # 多层感知机（MLP）的维度
    block_num = 20  # Transformer 块的数量
    patch_dim = 16  # 图像被分割成的小块的大小
    class_num = 3  # 类别数，对于二分类或单类别分割问题可能是1
    # 使用这些参数创建 TransUNet 类的实例
    model = TransUNet3plus(in_channels=1, out_channels=3, img_dim=img_dim, head_num=head_num, mlp_dim=mlp_dim,
                           block_num=block_num, patch_dim=patch_dim, class_num=class_num)
    device = "cuda"
    model.to(device)

    # print(model)
    cal_params_flops(model, img_dim)
    summary(model, input_size=(1, 256, 256))
