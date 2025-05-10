import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch


#from reflib.face.utils import CFaceUtils

class CFacePsringNormLayer(nn.Module):
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(CFacePsringNormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            assert 1 == 0, f'Norm type {norm_type} not support.'

    def forward(self, x, ref=None):
        if self.norm_type == 'spade':
            return self.norm(x, ref)
        else:
            return self.norm(x)

class CFacePsringReluLayer(nn.Module):

    def __init__(self, channels, relu_type='relu'):
        super(CFacePsringReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x * 1.0
        else:
            assert 1 == 0, f'Relu type {relu_type} not support.'

    def forward(self, x):
        return self.func(x)


class CFacePsringConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 scale='none',
                 norm_type='none',
                 relu_type='none',
                 use_pad=True,
                 bias=True):
        super(CFacePsringConvLayer, self).__init__()
        self.use_pad = use_pad
        self.norm_type = norm_type
        if norm_type in ['bn']:
            bias = False

        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(int(np.ceil((kernel_size - 1.) / 2)))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = CFacePsringReluLayer(out_channels, relu_type)
        self.norm = CFacePsringNormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class CFacePsringResidualBlock(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none'):
        super(CFacePsringResidualBlock, self).__init__()

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = CFacePsringConvLayer(c_in, c_out, 3, scale)

        scale_config_dict = {'down': ['none', 'down'], 'up': ['up', 'none'], 'none': ['none', 'none']}
        scale_conf = scale_config_dict[scale]

        self.conv1 = CFacePsringConvLayer(c_in, c_out, 3, scale_conf[0], norm_type=norm_type, relu_type=relu_type)
        self.conv2 = CFacePsringConvLayer(c_out, c_out, 3, scale_conf[1], norm_type=norm_type, relu_type='none')

    def forward(self, x):
        identity = self.shortcut_func(x)

        res = self.conv1(x)
        res = self.conv2(res)
        return identity + res


class CFacePsringParseNet(nn.Module):

    def __init__(self,
                 in_size=128,
                 out_size=128,
                 min_feat_size=32,
                 base_ch=64,
                 parsing_ch=19,
                 res_depth=10,
                 relu_type='LeakyReLU',
                 norm_type='bn',
                 ch_range=[32, 256]):
        super().__init__()
        self.res_depth = res_depth
        act_args = {'norm_type': norm_type, 'relu_type': relu_type}
        min_ch, max_ch = ch_range

        ch_clip = lambda x: max(min_ch, min(x, max_ch))  # noqa: E731
        min_feat_size = min(in_size, min_feat_size)

        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))

        # =============== define encoder-body-decoder ====================
        self.encoder = []
        self.encoder.append(CFacePsringConvLayer(3, base_ch, 3, 1))
        head_ch = base_ch
        for i in range(down_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch * 2)
            self.encoder.append(CFacePsringResidualBlock(cin, cout, scale='down', **act_args))
            head_ch = head_ch * 2

        self.body = []
        for i in range(res_depth):
            self.body.append(CFacePsringResidualBlock(ch_clip(head_ch), ch_clip(head_ch), **act_args))

        self.decoder = []
        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            self.decoder.append(CFacePsringResidualBlock(cin, cout, scale='up', **act_args))
            head_ch = head_ch // 2

        self.encoder = nn.Sequential(*self.encoder)
        self.body = nn.Sequential(*self.body)
        self.decoder = nn.Sequential(*self.decoder)
        self.out_img_conv = CFacePsringConvLayer(ch_clip(head_ch), 3)
        self.out_mask_conv = CFacePsringConvLayer(ch_clip(head_ch), parsing_ch)

    def forward(self, x):
        feat = self.encoder(x)
        x = feat + self.body(feat)
        x = self.decoder(x)
        out_img = self.out_img_conv(x)
        out_mask = self.out_mask_conv(x)
        return out_mask, out_img

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class CFacePsringBasicBlock(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1):
        super(CFacePsringBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class CFacePsringResNet18(nn.Module):

    def __init__(self):
        super(CFacePsringResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat8, feat16, feat32


class CFacePsringConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(CFacePsringConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class CFacePsringBiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, num_class):
        super(CFacePsringBiSeNetOutput, self).__init__()
        self.conv = CFacePsringConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, num_class, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.conv(x)
        out = self.conv_out(feat)
        return out, feat


class CFacePsringAttentionRefinementModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(CFacePsringAttentionRefinementModule, self).__init__()
        self.conv = CFacePsringConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class CFacePsringContextPath(nn.Module):

    def __init__(self):
        super(CFacePsringContextPath, self).__init__()
        self.resnet = CFacePsringResNet18()
        self.arm16 = CFacePsringAttentionRefinementModule(256, 128)
        self.arm32 = CFacePsringAttentionRefinementModule(512, 128)
        self.conv_head32 = CFacePsringConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = CFacePsringConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = CFacePsringConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

class CFacePsringFeatureFusionModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(CFacePsringFeatureFusionModule, self).__init__()
        self.convblk = CFacePsringConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class CFacePsringBiSeNet(nn.Module):

    def __init__(self, num_class):
        super(CFacePsringBiSeNet, self).__init__()
        self.cp = CFacePsringContextPath()
        self.ffm = CFacePsringFeatureFusionModule(256, 256)
        self.conv_out = CFacePsringBiSeNetOutput(256, 256, num_class)
        self.conv_out16 = CFacePsringBiSeNetOutput(128, 64, num_class)
        self.conv_out32 = CFacePsringBiSeNetOutput(128, 64, num_class)

    def forward(self, x, return_feat=False):
        h, w = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # return res3b1 feature
        feat_sp = feat_res8  # replace spatial path feature with res3b1 feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        out, feat = self.conv_out(feat_fuse)
        out16, feat16 = self.conv_out16(feat_cp8)
        out32, feat32 = self.conv_out32(feat_cp16)

        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        out16 = F.interpolate(out16, (h, w), mode='bilinear', align_corners=True)
        out32 = F.interpolate(out32, (h, w), mode='bilinear', align_corners=True)

        if return_feat:
            feat = F.interpolate(feat, (h, w), mode='bilinear', align_corners=True)
            feat16 = F.interpolate(feat16, (h, w), mode='bilinear', align_corners=True)
            feat32 = F.interpolate(feat32, (h, w), mode='bilinear', align_corners=True)
            return out, out16, out32, feat, feat16, feat32
        else:
            return out, out16, out32


class CFaceParsing:
    def __init__(self ):
        self.mydata = 0

    def init_parsing_model(self, model_name='bisenet', half=False, device='cuda'):
        model = CFacePsringParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_path = 'reflib/model/parsing_parsenet.pth'
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(load_net, strict=True)
        model.eval()
        model = model.to(device)
        return model
